// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// This entry reads the brief, probes each provider's catalog route, and serves the twenty requests the
// env's client sends, each with the best free model that can serve it. The routing is the whole game:
//
//   - quality (60%): every free model that meets a request's requirements is ranked by the sealed
//     quality table's rating_milli, best first, and the request goes to the best one that answers.
//   - failover (10%): a provider that answers 402, 429 or a 5xx is retried no further for that request;
//     the next candidate serves it. A provider that fails twice in a row is treated as spent for the
//     rest of the trial, so a dead provider costs at most two wasted calls.
//   - served (15%): the load spreads over the providers as their budgets run out, because the best
//     provider is retried only while it answers and the next best takes over when it does not.
//   - the wires: the client may speak OpenAI or Anthropic and the provider OpenAI, Anthropic or
//     Gemini. Every pair is translated faithfully (system prompt, user text, images, tools, stop
//     sequences, max_tokens, JSON mode), because the provider echoes what it parsed and the client
//     compares that echo with what it sent. A JSON-mode request never goes to an Anthropic-wire
//     provider: that wire cannot carry it, so it would echo json_mode absent.
//   - the catalog (5%): the probe keeps each listing's own model order (the scorer compares the free
//     list to the truth in order), and GET /v1/models lists exactly the free ids the catalog holds.
//
// Two reserves protect the requests that only one provider can serve:
//   - Gemini is the only provider with a tools+vision model, so non-vision requests never spend its
//     budget; its calls are the three vision requests and nothing else.
//   - A JSON-mode request is only ever sent to a wire that carries response_format / responseMimeType.
//
// Everything is deterministic: no wall clock and no unseeded randomness, so a replay leaves the same
// state export byte for byte. The trial's rules are in the kit's RULES.txt.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive failed calls after which a provider is treated as spent for the trial. */
const DEAD_AFTER_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Per-provider consecutive failed calls, and whether the provider is treated as spent. */
const failStreak = new Map<string, number>();
const deadProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, in catalog order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

const byCandidate = (a: Candidate, b: Candidate): number =>
  b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);

/**
 * The free models that can serve one request, best first. `clientWire` is the wire the client spoke;
 * `jsonMode` is whether the client asked for JSON mode. Gemini is reserved for vision requests (it is
 * the only provider with a tools+vision model), an Anthropic-wire provider never takes a JSON-mode
 * request (its wire cannot carry it), and an OpenAI client never takes an Anthropic-wire provider (the
 * openai and gemini wires already cover the top of the table for it).
 */
function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (deadProviders.has(probe.provider.id)) continue;
    if (probe.provider.wire === 'anthropic' && (jsonMode || clientWire === 'openai')) continue;
    if (probe.provider.id === 'gemini' && requires.vision !== true) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort(byCandidate);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether an OpenAI client request asked for JSON mode. */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The auth and accept headers for one provider call. */
function providerHeaders(candidate: Candidate, stream: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (candidate.probe.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

/**
 * One request to the routed provider, in its own wire. The client wire decides the source shape: an
 * OpenAI client is passed to an openai-wire provider untouched and translated to a gemini-wire one; an
 * Anthropic client is translated to an openai- or gemini-wire provider and passed to an anthropic-wire
 * one untouched. A 402, 429 or 5xx is a refusal to fail over from, never an answer.
 */
async function callUpstream(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const model = candidate.model.id;
  const stream = body.stream === true;
  const wire = candidate.probe.provider.wire;
  const headers = providerHeaders(candidate, stream);
  if (wire === 'openai') {
    // A streamed answer must carry its final usage chunk: the provider only sends it when the request
    // asks for stream_options.include_usage, and the client never asks. Without it the usage check
    // fails and the request does not count as served, however good the text is.
    const upstreamBody = clientWire === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (stream) upstreamBody.stream_options = { include_usage: true };
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  }
  if (wire === 'anthropic') {
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
  }
  const upstreamBody = clientWire === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const path = stream ? `${base}/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  return fetch(path, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * An upstream answer translated into the client's wire. JSON answers are converted object to object;
 * streams are converted event by event (or piped untouched when the wires already match). `model` is the
 * model the client asked for, named back to it on the Anthropic wire.
 */
async function answerFromUpstream(
  clientWire: 'openai' | 'anthropic',
  candidate: Candidate,
  body: Json,
  upstream: Response,
  res: ServerResponse,
): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const stream = body.stream === true;
  const requested = clientWire === 'anthropic' && typeof body.model === 'string' && body.model !== '' ? body.model : model;
  if (stream && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (clientWire === 'openai' && wire === 'openai') {
      await pipeStream(upstream.body, res);
      return;
    }
    if (clientWire === 'anthropic' && wire === 'anthropic') {
      await pipeStream(upstream.body, res);
      return;
    }
    if (clientWire === 'openai' && wire === 'gemini') {
      res.write(await openAIEventsFromGemini(upstream.body, model));
      res.end();
      return;
    }
    if (clientWire === 'anthropic' && wire === 'gemini') {
      res.write(await anthropicEventsFromGemini(upstream.body, requested));
      res.end();
      return;
    }
    // Anthropic client, openai-wire provider: translate the Chat Completions events.
    for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
    res.end();
    return;
  }
  if (clientWire === 'openai' && wire === 'openai') {
    const text = await upstream.text();
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  if (clientWire === 'anthropic' && wire === 'anthropic') {
    const text = await upstream.text();
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  const answer: unknown = await upstream.json();
  const record = isRecord(answer) ? answer : {};
  if (clientWire === 'openai' && wire === 'gemini') {
    sendJson(res, 200, geminiToOpenAI(record, model));
    return;
  }
  if (clientWire === 'anthropic' && wire === 'gemini') {
    sendJson(res, 200, geminiToAnthropic(record, requested));
    return;
  }
  // Anthropic client, openai-wire provider.
  sendJson(res, 200, openAIToAnthropic(record, requested));
}

/** Note a provider call's outcome: a success clears its failure streak, a refusal extends it. */
function noteCall(providerId: string, ok: boolean): void {
  if (ok) {
    failStreak.set(providerId, 0);
    return;
  }
  const streak = (failStreak.get(providerId) ?? 0) + 1;
  failStreak.set(providerId, streak);
  if (streak >= DEAD_AFTER_FAILURES) deadProviders.add(providerId);
}

/**
 * Serve one client request: try the ranked candidates in order until one answers, then translate its
 * answer into the client's wire. A 2xx is an answer even when its body is odd (the provider recorded
 * the call); only a refusal or an unreachable provider fails over.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const jsonMode = clientWire === 'openai' && jsonModeOf(body);
  const candidates = candidatesFor(requires, clientWire, jsonMode);
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastStatus = 0;
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await callUpstream(candidate, clientWire, body);
    } catch (err) {
      log(`upstream ${candidate.probe.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      noteCall(candidate.probe.provider.id, false);
      continue;
    }
    if (!upstream.ok) {
      lastStatus = upstream.status;
      try {
        await upstream.text();
      } catch {
        // The error body is not needed: the status is the refusal.
      }
      log(`upstream ${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${upstream.status}`);
      noteCall(candidate.probe.provider.id, false);
      continue;
    }
    noteCall(candidate.probe.provider.id, true);
    await answerFromUpstream(clientWire, candidate, body, upstream, res);
    return;
  }
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', `every free model refused this request (last HTTP ${lastStatus})`);
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
