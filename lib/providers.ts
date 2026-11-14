// How a trial's brief is read, and how the catalog is probed.
//
// The brief (the file MAKEFUN_PROMPT_FILE names) holds three JSON documents
// buried in prose; this file pulls each one out by its `schema` string:
//
//   - the provider directory (`free-model-aggregator-01/providers/v1`): a row
//     per provider — its id, its wire, how it takes a key, the environment
//     variable a real deployment would read the key from, the key the
//     simulated service accepts, and the base path the entry joins onto
//     MAKEFUN_ENV_URL. A directory row may spell an absolute base_url
//     instead of a path; joinBase accepts either.
//   - the quality table (`free-model-aggregator-01/ratings/v1`): every
//     model's public rating in milli-points, what it can do (tools, vision,
//     JSON mode, context length) and whether it is free. The scorer ranks an
//     answer by rating; a paid model is never a legal pick. This baseline
//     reads the table's capability columns and free flag (the sealed truth,
//     where a row exists) but never its rating (`--list` shows why in
//     main.ts's header, and the routing TODO there names the gap).
//   - the workload manifest (`free-model-aggregator-01/workload/v1`): the
//     requests the client will send, in order, each with its api, whether it
//     streams, its weight, and the requirements a model must meet to serve it.
//
// Each provider's catalog route is then probed in the provider's own wire:
//
//   openai     GET <base>/v1/models          Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models          x-api-key: <key>, anthropic-version: 2023-06-01
//   gemini     GET <base>/v1beta/models      x-goog-api-key: <key>
//
// A probe that fails — a refused connection, a non-2xx status, a body that is
// not JSON — leaves its provider out of the catalog and the provider gets no
// traffic; the trial goes on. Nothing here depends on the clock or on chance,
// so `node main.ts --list` and the body posted to /register are the same
// document (main.ts prints or posts this one object).

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief lists it. `base` is the base path (or absolute URL) its routes hang off. */
export interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderDirectoryEntry[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** One row of the sealed quality table (the brief's copy). */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** What one request needs: a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

/** The brief as the baseline reads it. `error` is set when it could not be read at all. */
export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered, and the models it listed. */
export interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** What a model can do, from the listing first and the quality table second (null: neither said). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** The first of the given strings that is not empty, or '' when all of them are. */
function firstNonEmpty(...values: string[]): string {
  for (const value of values) {
    if (value !== '') return value;
  }
  return '';
}

/** The first of the given values that is neither null nor undefined, or null when there is none. */
function prefer<T>(...values: (T | null | undefined)[]): T | null {
  for (const value of values) {
    if (value !== null && value !== undefined) return value;
  }
  return null;
}

/** Whether `v` is exactly the boolean true (anything else is not a yes). */
function isTrue(v: unknown): boolean {
  return v === true;
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return stripTrailingSlashes(trimmed);
  const prefix = stripTrailingSlashes(envUrl);
  if (trimmed === '') return prefix;
  const path = stripTrailingSlashes(trimmed);
  return `${prefix}${path.startsWith('/') ? '' : '/'}${path}`;
}

/** `s` with every trailing slash removed. */
function stripTrailingSlashes(s: string): string {
  return s.replace(/\/+$/, '');
}

/** The header name a provider's auth scheme puts its key under, if it uses one. */
const AUTH_HEADER_NAME: Record<AuthScheme, string | undefined> = {
  bearer: 'authorization',
  'x-api-key': 'x-api-key',
  'x-goog-api-key': 'x-goog-api-key',
  none: undefined,
};

/** The headers a provider wants its key in. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  const name = AUTH_HEADER_NAME[auth];
  if (name === undefined) return {};
  return { [name]: auth === 'bearer' ? `Bearer ${key}` : key };
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const found: Json[] = [];
  let start = text.indexOf('{');
  while (start !== -1) {
    const end = objectEnd(text, start);
    if (end !== -1) {
      try {
        const value: unknown = JSON.parse(text.slice(start, end + 1));
        if (isRecord(value) && typeof value.schema === 'string') found.push(value);
      } catch {
        // Not JSON on its own: look on.
      }
    }
    start = text.indexOf('{', start + 1);
  }
  return found;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      i = skipQuoted(text, i);
      continue;
    }
    if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/**
 * The index of the closing quote of the string that opens at `i`, or the last
 * index when the string never closes. A backslash skips the character after
 * it, so an escaped quote does not end the string.
 */
function skipQuoted(text: string, i: number): number {
  let j = i + 1;
  while (j < text.length) {
    const ch = text[j];
    if (ch === '\\') {
      j += 2;
      continue;
    }
    if (ch === '"') return j;
    j += 1;
  }
  return text.length - 1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

const WIRES = ['openai', 'anthropic', 'gemini'] as const;
const AUTH_SCHEMES = ['bearer', 'x-api-key', 'x-goog-api-key', 'none'] as const;

function isWire(v: unknown): v is Wire {
  return typeof v === 'string' && (WIRES as readonly string[]).includes(v);
}

function isAuthScheme(v: unknown): v is AuthScheme {
  return typeof v === 'string' && (AUTH_SCHEMES as readonly string[]).includes(v);
}

function providerOf(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = isWire(v.wire) ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme = isAuthScheme(v.auth) ? v.auth : 'bearer';
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a directory that spells an
  // absolute base instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const base = firstNonEmpty(stringOf(v.path), stringOf(v.base_path), stringOf(v.base_url), stringOf(v.base));
  return {
    id,
    name: firstNonEmpty(stringOf(v.name), id),
    wire,
    auth,
    base,
    key_env: stringOf(v.key_env),
    key: stringOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: isTrue(v.tools),
    vision: isTrue(v.vision),
    json: isTrue(v.json),
    free: isTrue(v.free),
  };
}

/** The requirements one workload row states; a row that says nothing needs nothing beyond the defaults. */
function requirementOf(v: unknown): Requirement {
  const req = isRecord(v) ? v : {};
  const out: Requirement = {
    tools: isTrue(req.tools),
    vision: isTrue(req.vision),
    json: isTrue(req.json),
  };
  const minContext = numberOrNull(req.min_context);
  if (minContext !== null) out.min_context = minContext;
  return out;
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: isTrue(v.stream),
    weight: v.weight === 2 ? 2 : 1,
    requires: requirementOf(v.requires),
  };
}

/** The first document with this `schema` whose `listKey` member is an array, or null. */
function findDocument(docs: Json[], schema: string, listKey: string): Json | null {
  return docs.find((d) => d.schema === schema && Array.isArray(d[listKey])) ?? null;
}

/** The quality table as a RatingsFile, from the document that carries it. */
function parseRatingsDocument(doc: Json): RatingsFile {
  const rows = (doc.ratings as unknown[]).flatMap((v) => {
    const row = ratingOf(v);
    return row === null ? [] : [row];
  });
  const best: { id: string; rating_milli: number }[] = Array.isArray(doc.best)
    ? doc.best.flatMap((v) => {
        if (!isRecord(v) || stringOf(v.id) === '') return [];
        return [{ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 }];
      })
    : [];
  return { schema: RATINGS_SCHEMA, ratings: rows, best };
}

/** The workload manifest as a WorkloadFile, from the document that carries it. */
function parseWorkloadDocument(doc: Json): WorkloadFile {
  return {
    schema: WORKLOAD_SCHEMA,
    manifest: (doc.manifest as unknown[]).flatMap((v) => {
      const request = workloadRequestOf(v);
      return request === null ? [] : [request];
    }),
  };
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  const directory: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (path === '') return { directory, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = findDocument(docs, PROVIDERS_SCHEMA, 'providers');
  if (directoryDoc === null) {
    return { directory, ratings: null, workload: null, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = (directoryDoc.providers as unknown[]).flatMap((v) => {
    const entry = providerOf(v);
    return entry === null ? [] : [entry];
  });
  const ratingsDoc = findDocument(docs, RATINGS_SCHEMA, 'ratings');
  const ratings = ratingsDoc === null ? null : parseRatingsDocument(ratingsDoc);
  const workloadDoc = findDocument(docs, WORKLOAD_SCHEMA, 'manifest');
  const workload = workloadDoc === null ? null : parseWorkloadDocument(workloadDoc);
  return { directory: { schema: PROVIDERS_SCHEMA, providers }, ratings, workload, error: null };
}

/** The rating row of one provider's model, when the quality table has one. */
export function ratingOfModel(ratings: RatingsFile | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (it is the sealed truth), the
 * listing's own word where it does not, and paid when neither says. A paid model is never routed.
 */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's own fields where it does not. */
export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: prefer(row?.context_length, model.context_length),
    tools: prefer(row?.tools, model.tools),
    vision: prefer(row?.vision, model.vision),
    json: prefer(row?.json, model.json),
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  const capabilityChecks: [needed: boolean | undefined, actual: boolean | null][] = [
    [requires.tools, caps.tools],
    [requires.vision, caps.vision],
    [requires.json, caps.json],
  ];
  for (const [needed, actual] of capabilityChecks) {
    if (needed === true && actual !== true) return false;
  }
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** The numeric values a listing's `pricing` member states (strings are read as numbers). */
function priceValues(pricing: Json): number[] {
  return Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
}

/** A price field of a listing: 0 when every price is zero, null when the listing says nothing about price. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = priceValues(pricing);
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** The name a listing gives a model, falling back through the wire's spellings to the id itself. */
function modelDisplayName(v: Json, fallback: string): string {
  return firstNonEmpty(stringOf(v.display_name), stringOf(v.displayName), fallback);
}

/** A listing's context length, under whichever of the wires' names it uses. */
function modelContextLength(v: Json): number | null {
  return prefer(numberOrNull(v.context_length), numberOrNull(v.context_window), numberOrNull(v.inputTokenLimit));
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  return {
    id,
    display_name: modelDisplayName(v, id),
    free: prefer(boolOrNull(v.free), zeroPriced(v)),
    context_length: modelContextLength(v),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const model = modelIn(wire, v);
    return model === null ? [] : [model];
  });
}

/** The URL and headers one provider's catalog probe uses, in its own wire. */
function probeTarget(envUrl: string, provider: ProviderDirectoryEntry): { url: string; headers: Record<string, string> } {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return { url, headers };
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. Retries once: a
 * missing provider costs the catalog its four checks, and a slow first probe is not worth that. */
export async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry, timeoutMs = 5000): Promise<Probe> {
  const { url, headers } = probeTarget(envUrl, provider);
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
      if (!res.ok) {
        if (attempt === 0) continue;
        return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
      }
      const body: unknown = await res.json();
      return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
    } catch (err) {
      if (attempt === 0) continue;
      return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
    }
  }
  return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed` };
}

/**
 * The models one probed provider contributes to the catalog, in the listing's
 * own order and with each id once. The listing's order is kept, never sorted:
 * the scorer compares the free-model list to the truth in order, and the truth
 * lists them in the provider's own order (the order the catalog route gave).
 */
function catalogModelsFor(probe: Probe, ratings: RatingsFile | null): CatalogModel[] {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of probe.models) {
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    const row = ratingOfModel(ratings, probe.provider.id, model.id);
    models.push({ id: model.id, free: modelFree(model, row), context_length: prefer(row?.context_length, model.context_length) });
  }
  return models;
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every model it listed.
 *
 * Its base_url is the directory's own path (`/p/<id>`), never the path joined to `envUrl`: the env's
 * origin differs between a live pass and its replay (a fresh loopback port under the process driver, a
 * fresh pod on the k8s layout), and the catalog is part of the state export a replay must reproduce byte
 * for byte. The scorer compares a reported base_url by its path suffix, so the path passes exactly as the
 * joined URL does; the joined URL reaches a provider, the path names it.
 */
export function buildCatalog(_envUrl: string, probes: readonly Probe[], ratings: RatingsFile | null): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models: catalogModelsFor(probe, ratings),
    });
  }
  return { providers };
}
