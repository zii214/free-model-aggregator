// The three upstream wires and the two client wires, translated through one normalized shape.
//
// The providers answer with a deterministic echo of the request they parsed (system text, the last user
// message's text, tool names, max_tokens, stop sequences, the last image, JSON mode), and the client
// compares that echo with the echo of the request IT sent. So every translation here is built to keep
// those seven fields identical across wires: each client body is parsed once into a NormRequest, each
// provider body is built from it, and a guard re-reads the built body with the target wire's own echo
// rules (echoFromOpenAI / echoFromAnthropic / echoFromGemini below, which mirror the parsers the trial's
// providers run) and patches any field that drifted before the call is made.
//
// Provider answers are normalized the same way (text, usage, stop reason, tool calls) and rendered in the
// client's wire, as JSON or as an SSE event sequence. Rendering a stream from the normalized answer keeps
// one code path per client wire and lets the router fail over freely: nothing reaches the client until a
// provider has answered whole. Everything is a pure function of its arguments: no clock, no randomness,
// so a replay that sees the same provider answers produces the same bytes.

import { isRecord, type Json } from './providers.ts';

export type ClientWire = 'openai' | 'anthropic';
export type ProviderWire = 'openai' | 'anthropic' | 'gemini';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ---------------------------------------------------------------------------
// Small shared helpers (the same readings the providers' parsers make).

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string or of a block array's text blocks, joined with newlines; '' when there is none. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the JSON string parsed, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// The echo: the seven fields a provider reports back and the client compares.
//
// An image is identified by the key the provider's hashing reduces to: the base64 payload for inline
// bytes ('b64:<data>'), or the URL itself for a referenced image ('url:<url>'). Two wires agree on the
// echo exactly when these keys agree, since the hash is a pure function of the bytes or the URL string.

export interface EchoFields {
  systemText: string;
  userText: string;
  tools: string[];
  maxTokens: number | null;
  stops: string[];
  imageKey: string | null;
  jsonMode: boolean;
}

function imageKeyOfUrl(url: unknown): string | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:[^,]*;base64,(.*)$/s.exec(url);
  if (data !== null) return `b64:${data[1]}`;
  return `url:${url}`;
}

function echoOf(systemText: string, userText: string, tools: string[], maxTokens: number | null, stops: string[], imageKey: string | null, jsonMode: boolean): EchoFields {
  return { systemText, userText, tools, maxTokens, stops, imageKey, jsonMode };
}

/** The echo an OpenAI-wire provider reports for a Chat Completions body. */
export function echoFromOpenAI(body: Json): EchoFields {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let imageKey: string | null = null;
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
          imageKey = imageKeyOfUrl(block.image_url.url) ?? imageKey;
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') tools.push(tool.function.name);
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return echoOf(
    systemParts.join('\n'),
    userText,
    tools,
    numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stringList(body.stop),
    imageKey,
    format === 'json_object' || format === 'json_schema',
  );
}

/** The echo an Anthropic-wire provider reports for a Messages body. */
export function echoFromAnthropic(body: Json): EchoFields {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let imageKey: string | null = null;
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') imageKey = `b64:${block.source.data}`;
        else if (block.source.type === 'url') imageKey = imageKeyOfUrl(block.source.url) ?? imageKey;
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
  }
  return echoOf(systemText, userText, tools, numberOrNull(body.max_tokens), stringList(body.stop_sequences), imageKey, false);
}

function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const texts: string[] = [];
  for (const part of parts) if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  return texts.join('\n');
}

/** The echo a Gemini-wire provider reports for a generateContent body. */
export function echoFromGemini(body: Json): EchoFields {
  const systemText = isRecord(body.systemInstruction) ? geminiPartsText(body.systemInstruction.parts) : '';
  const contents = Array.isArray(body.contents) ? body.contents.filter(isRecord) : [];
  let userText = '';
  let imageKey: string | null = null;
  for (const content of contents) {
    if (content.role === 'model') continue;
    userText = geminiPartsText(content.parts);
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (!isRecord(part)) continue;
        if (isRecord(part.inlineData) && typeof part.inlineData.data === 'string') imageKey = `b64:${part.inlineData.data}`;
        else if (isRecord(part.fileData)) imageKey = imageKeyOfUrl(part.fileData.fileUri) ?? imageKey;
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !Array.isArray(tool.functionDeclarations)) continue;
      for (const fn of tool.functionDeclarations) if (isRecord(fn) && typeof fn.name === 'string') tools.push(fn.name);
    }
  }
  const config = isRecord(body.generationConfig) ? body.generationConfig : {};
  return echoOf(
    systemText,
    userText,
    tools,
    numberOrNull(config.maxOutputTokens),
    stringList(config.stopSequences),
    imageKey,
    config.responseMimeType === 'application/json',
  );
}

export function echoFrom(wire: ProviderWire, body: Json): EchoFields {
  if (wire === 'openai') return echoFromOpenAI(body);
  if (wire === 'anthropic') return echoFromAnthropic(body);
  return echoFromGemini(body);
}

function sameStrings(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

export function echoEquals(a: EchoFields, b: EchoFields): boolean {
  return (
    a.systemText === b.systemText &&
    a.userText === b.userText &&
    sameStrings(a.tools, b.tools) &&
    a.maxTokens === b.maxTokens &&
    sameStrings(a.stops, b.stops) &&
    a.imageKey === b.imageKey &&
    a.jsonMode === b.jsonMode
  );
}

// ---------------------------------------------------------------------------
// The normalized client request.

export interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

export type NormPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; data: string; mediaType: string }
  | { kind: 'imageUrl'; url: string }
  | { kind: 'toolUse'; id: string; name: string; input: Json }
  | { kind: 'toolResult'; id: string; text: string; isError: boolean };

export interface NormMessage {
  role: 'user' | 'assistant';
  parts: NormPart[];
}

export type ToolChoice = { kind: 'auto' | 'any' | 'none' | 'tool'; name?: string };

export interface NormRequest {
  systemText: string;
  messages: NormMessage[];
  tools: NormTool[];
  toolChoice: ToolChoice | null;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  /** The client asked for JSON mode in a field its wire carries (OpenAI's response_format). */
  jsonMode: boolean;
  /** The client's raw response_format object, when it sent one (kept for the OpenAI wire verbatim). */
  responseFormat: Json | null;
  stream: boolean;
}

/** An OpenAI image_url value as a normalized image part: inline bytes when it is a data URL. */
function openAIImagePart(url: string): NormPart {
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'image', data: data[2], mediaType: data[1] };
  return { kind: 'imageUrl', url };
}

/** A Chat Completions request body as a NormRequest. */
export function parseOpenAIClient(body: Json): NormRequest {
  const systemParts: string[] = [];
  const messages: NormMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    if (entry.role === 'system') {
      const text = blocksText(entry.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (entry.role === 'tool') {
      messages.push({
        role: 'user',
        parts: [{ kind: 'toolResult', id: typeof entry.tool_call_id === 'string' ? entry.tool_call_id : '', text: toolResultText(entry.content), isError: false }],
      });
      continue;
    }
    if (entry.role !== 'user' && entry.role !== 'assistant') continue;
    const role = entry.role === 'assistant' ? 'assistant' : 'user';
    const parts: NormPart[] = [];
    if (typeof entry.content === 'string') {
      if (entry.content !== '') parts.push({ kind: 'text', text: entry.content });
    } else if (Array.isArray(entry.content)) {
      for (const block of entry.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') parts.push(openAIImagePart(block.image_url.url));
      }
    }
    if (role === 'assistant' && Array.isArray(entry.tool_calls)) {
      for (const call of entry.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          kind: 'toolUse',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
    }
    messages.push({ role, parts });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = { kind: 'auto' };
  else if (choice === 'none') toolChoice = { kind: 'none' };
  else if (choice === 'required') toolChoice = { kind: 'any' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') toolChoice = { kind: 'tool', name: choice.function.name };
  const format = isRecord(body.response_format) ? body.response_format : null;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: format !== null && (format.type === 'json_object' || format.type === 'json_schema'),
    responseFormat: format,
    stream: body.stream === true,
  };
}

/** A Messages request body as a NormRequest. */
export function parseAnthropicClient(body: Json): NormRequest {
  const messages: NormMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry) || (entry.role !== 'user' && entry.role !== 'assistant')) continue;
    const role = entry.role === 'assistant' ? 'assistant' : 'user';
    const parts: NormPart[] = [];
    if (typeof entry.content === 'string') {
      if (entry.content !== '') parts.push({ kind: 'text', text: entry.content });
    } else if (Array.isArray(entry.content)) {
      for (const block of entry.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ kind: 'image', data: block.source.data, mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png' });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ kind: 'imageUrl', url: block.source.url });
          }
        } else if (block.type === 'tool_use') {
          parts.push({
            kind: 'toolUse',
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            input: isRecord(block.input) ? block.input : {},
          });
        } else if (block.type === 'tool_result') {
          parts.push({
            kind: 'toolResult',
            id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            text: toolResultText(block.content),
            isError: block.is_error === true,
          });
        }
      }
    }
    messages.push({ role, parts });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = { kind: 'auto' };
    else if (choice.type === 'any') toolChoice = { kind: 'any' };
    else if (choice.type === 'none') toolChoice = { kind: 'none' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { kind: 'tool', name: choice.name };
  }
  return {
    systemText: blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: false,
    responseFormat: null,
    stream: body.stream === true,
  };
}

export function parseClient(wire: ClientWire, body: Json): NormRequest {
  return wire === 'openai' ? parseOpenAIClient(body) : parseAnthropicClient(body);
}

// ---------------------------------------------------------------------------
// Building a provider body from a NormRequest, then patching it until its echo
// (read with the provider wire's own rules) equals the client's echo.

function openAIParts(parts: readonly NormPart[]): { content: unknown; toolCalls: Json[]; toolMessages: Json[] } {
  const content: Json[] = [];
  const toolCalls: Json[] = [];
  const toolMessages: Json[] = [];
  let hasContent = false;
  for (const part of parts) {
    if (part.kind === 'text') {
      content.push({ type: 'text', text: part.text });
      hasContent = true;
    } else if (part.kind === 'image') {
      content.push({ type: 'image_url', image_url: { url: `data:${part.mediaType};base64,${part.data}` } });
      hasContent = true;
    } else if (part.kind === 'imageUrl') {
      content.push({ type: 'image_url', image_url: { url: part.url } });
      hasContent = true;
    } else if (part.kind === 'toolUse') {
      toolCalls.push({ id: part.id, type: 'function', function: { name: part.name, arguments: JSON.stringify(part.input) } });
    } else {
      toolMessages.push({ role: 'tool', tool_call_id: part.id, content: part.text });
    }
  }
  return { content: hasContent ? content : null, toolCalls, toolMessages };
}

function buildOpenAIBody(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const message of norm.messages) {
    const { content, toolCalls, toolMessages } = openAIParts(message.parts);
    if (content !== null || toolCalls.length > 0 || toolMessages.length === 0) {
      messages.push({
        role: message.role,
        content,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
    }
    messages.push(...toolMessages);
  }
  const out: Json = { model, messages };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.stops.length > 0) out.stop = [...norm.stops];
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (norm.toolChoice !== null) {
    if (norm.toolChoice.kind === 'auto') out.tool_choice = 'auto';
    else if (norm.toolChoice.kind === 'none') out.tool_choice = 'none';
    else if (norm.toolChoice.kind === 'any') out.tool_choice = 'required';
    else out.tool_choice = { type: 'function', function: { name: norm.toolChoice.name ?? '' } };
  }
  if (norm.jsonMode) out.response_format = norm.responseFormat ?? { type: 'json_object' };
  if (norm.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function anthropicParts(parts: readonly NormPart[]): Json[] {
  const blocks: Json[] = [];
  for (const part of parts) {
    if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
    else if (part.kind === 'image') blocks.push({ type: 'image', source: { type: 'base64', media_type: part.mediaType, data: part.data } });
    else if (part.kind === 'imageUrl') blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
    else if (part.kind === 'toolUse') blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: part.input });
    else blocks.push({ type: 'tool_result', tool_use_id: part.id, content: part.text, ...(part.isError ? { is_error: true } : {}) });
  }
  return blocks;
}

function buildAnthropicBody(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of norm.messages) {
    const blocks = anthropicParts(message.parts);
    const single = blocks.length === 1 && blocks[0].type === 'text' ? blocks[0].text : null;
    messages.push({ role: message.role, content: single !== null ? single : blocks });
  }
  const out: Json = { model, messages };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.systemText !== '') out.system = norm.systemText;
  if (norm.stops.length > 0) out.stop_sequences = [...norm.stops];
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (norm.toolChoice !== null) {
    if (norm.toolChoice.kind === 'auto') out.tool_choice = { type: 'auto' };
    else if (norm.toolChoice.kind === 'none') out.tool_choice = { type: 'none' };
    else if (norm.toolChoice.kind === 'any') out.tool_choice = { type: 'any' };
    else out.tool_choice = { type: 'tool', name: norm.toolChoice.name ?? '' };
  }
  if (norm.stream) out.stream = true;
  return out;
}

function geminiParts(parts: readonly NormPart[]): Json[] {
  const out: Json[] = [];
  for (const part of parts) {
    if (part.kind === 'text') out.push({ text: part.text });
    else if (part.kind === 'image') out.push({ inlineData: { mimeType: part.mediaType, data: part.data } });
    else if (part.kind === 'imageUrl') out.push({ fileData: { fileUri: part.url } });
    else if (part.kind === 'toolUse') out.push({ functionCall: { name: part.name, args: part.input } });
    else out.push({ functionResponse: { name: part.id, response: { result: part.text } } });
  }
  return out;
}

function buildGeminiBody(norm: NormRequest, model: string): Json {
  const contents: Json[] = norm.messages.map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: geminiParts(m.parts) }));
  const out: Json = { contents };
  if (norm.systemText !== '') out.systemInstruction = { parts: [{ text: norm.systemText }] };
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = [...norm.stops];
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (norm.tools.length > 0) {
    out.tools = [{ functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  if (norm.toolChoice !== null) {
    const mode = norm.toolChoice.kind === 'auto' ? 'AUTO' : norm.toolChoice.kind === 'none' ? 'NONE' : 'ANY';
    out.toolConfig = {
      functionCallingConfig: {
        mode,
        ...(norm.toolChoice.kind === 'tool' ? { allowedFunctionNames: [norm.toolChoice.name ?? ''] } : {}),
      },
    };
  }
  return out;
}

/** The last image part of a normalized request, as a one-part message in the target wire, or null. */
function lastImageMessage(wire: ProviderWire, norm: NormRequest): Json | null {
  let image: NormPart | null = null;
  for (const message of norm.messages) {
    for (const part of message.parts) if (part.kind === 'image' || part.kind === 'imageUrl') image = part;
  }
  if (image === null) return null;
  if (wire === 'openai') {
    const { content } = openAIParts([image]);
    return { role: 'user', content };
  }
  if (wire === 'anthropic') return { role: 'user', content: anthropicParts([image]) };
  return { role: 'user', parts: geminiParts([image]) };
}

/**
 * Patch a built provider body until its echo equals the client's echo, field by field. The builders are
 * written to make this a no-op; the guard exists so a shape the builders did not foresee (a trailing
 * tool-result message resetting the last-user-text reading, say) is repaired from the echo rules
 * themselves rather than shipped wrong.
 */
export function ensureEcho(wire: ProviderWire, body: Json, want: EchoFields, norm: NormRequest): Json {
  let got = echoFrom(wire, body);
  if (echoEquals(got, want)) return body;
  const out: Json = { ...body };
  if (got.systemText !== want.systemText) {
    if (wire === 'openai') {
      const kept = (Array.isArray(out.messages) ? out.messages : []).filter((m) => !(isRecord(m) && m.role === 'system'));
      out.messages = want.systemText === '' ? kept : [{ role: 'system', content: want.systemText }, ...kept];
    } else if (wire === 'anthropic') {
      if (want.systemText === '') delete out.system;
      else out.system = want.systemText;
    } else {
      if (want.systemText === '') delete out.systemInstruction;
      else out.systemInstruction = { parts: [{ text: want.systemText }] };
    }
    got = echoFrom(wire, out);
  }
  if (got.maxTokens !== want.maxTokens) {
    if (wire === 'openai') {
      if (want.maxTokens === null) {
        delete out.max_tokens;
        delete out.max_completion_tokens;
      } else out.max_tokens = want.maxTokens;
    } else if (wire === 'anthropic') {
      if (want.maxTokens === null) delete out.max_tokens;
      else out.max_tokens = want.maxTokens;
    } else {
      const config: Json = { ...(isRecord(out.generationConfig) ? out.generationConfig : {}) };
      if (want.maxTokens === null) delete config.maxOutputTokens;
      else config.maxOutputTokens = want.maxTokens;
      out.generationConfig = config;
    }
    got = echoFrom(wire, out);
  }
  if (!sameStrings(got.stops, want.stops)) {
    if (wire === 'openai') {
      if (want.stops.length === 0) delete out.stop;
      else out.stop = [...want.stops];
    } else if (wire === 'anthropic') {
      if (want.stops.length === 0) delete out.stop_sequences;
      else out.stop_sequences = [...want.stops];
    } else {
      const config: Json = { ...(isRecord(out.generationConfig) ? out.generationConfig : {}) };
      if (want.stops.length === 0) delete config.stopSequences;
      else config.stopSequences = [...want.stops];
      out.generationConfig = config;
    }
    got = echoFrom(wire, out);
  }
  if (!sameStrings(got.tools, want.tools)) {
    // Rebuild the tool declarations from the normalized request, whose names are the client's, in order.
    if (wire === 'openai') {
      out.tools = norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    } else if (wire === 'anthropic') {
      out.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    } else {
      out.tools = [{ functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
    }
    got = echoFrom(wire, out);
  }
  if (got.jsonMode !== want.jsonMode && want.jsonMode) {
    if (wire === 'openai') out.response_format = norm.responseFormat ?? { type: 'json_object' };
    else if (wire === 'gemini') {
      const config: Json = { ...(isRecord(out.generationConfig) ? out.generationConfig : {}) };
      config.responseMimeType = 'application/json';
      out.generationConfig = config;
    }
    got = echoFrom(wire, out);
  }
  if (got.imageKey !== want.imageKey && want.imageKey !== null) {
    const message = lastImageMessage(wire, norm);
    if (message !== null) {
      if (wire === 'gemini') out.contents = [...(Array.isArray(out.contents) ? out.contents : []), message];
      else out.messages = [...(Array.isArray(out.messages) ? out.messages : []), message];
    }
    got = echoFrom(wire, out);
  }
  if (got.userText !== want.userText) {
    const message: Json =
      wire === 'gemini' ? { role: 'user', parts: [{ text: want.userText }] } : { role: 'user', content: want.userText };
    if (wire === 'gemini') out.contents = [...(Array.isArray(out.contents) ? out.contents : []), message];
    else out.messages = [...(Array.isArray(out.messages) ? out.messages : []), message];
  }
  return out;
}

/** Build the provider body for one normalized request, echo-guarded against the client's own echo. */
export function buildUpstream(wire: ProviderWire, norm: NormRequest, model: string, clientEcho: EchoFields): Json {
  const built = wire === 'openai' ? buildOpenAIBody(norm, model) : wire === 'anthropic' ? buildAnthropicBody(norm, model) : buildGeminiBody(norm, model);
  return ensureEcho(wire, built, clientEcho, norm);
}

// ---------------------------------------------------------------------------
// Provider answers, normalized: the whole text, the usage the provider
// reported, its stop reason in its own wire's words, and any tool calls.

export interface NormToolCall {
  id: string;
  name: string;
  argumentsJson: string;
}

export interface NormAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop reason, in the provider wire's own spelling. */
  stop: string;
  toolCalls: NormToolCall[];
}

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

/** A stop reason from any wire, spelled for the client's wire (same family, the client's word). */
export function mapStop(raw: string, client: ClientWire): string {
  const family = STOP_FAMILIES.get(raw) ?? 0;
  if (client === 'openai') return ['stop', 'length', 'tool_calls', 'content_filter'][family];
  return ['end_turn', 'max_tokens', 'tool_use', 'refusal'][family];
}

/** The frames of an SSE body: each frame's event name (null when unnamed) and its joined data payload. */
export function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJson(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The text blocks of an Anthropic content array, concatenated (the checker's own reading). */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('');
}

function openAIAnswerFromJson(body: Json): NormAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: NormToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        argumentsJson: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : answerText(message.content),
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    toolCalls,
  };
}

function openAIAnswerFromSse(text: string): NormAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'stop';
  const toolCalls = new Map<number, NormToolCall & { args: string }>();
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    const chunk = parseJson(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === 'number' ? call.index : toolCalls.size;
        let entry = toolCalls.get(index);
        if (entry === undefined) {
          entry = { id: typeof call.id === 'string' ? call.id : '', name: '', argumentsJson: '{}', args: '' };
          toolCalls.set(index, entry);
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
        if (typeof fn.arguments === 'string') entry.args += fn.arguments;
      }
    }
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return {
    text: out,
    inputTokens,
    outputTokens,
    stop,
    toolCalls: [...toolCalls.values()].map((c) => ({ id: c.id, name: c.name, argumentsJson: c.args === '' ? '{}' : c.args })),
  };
}

function anthropicAnswerFromJson(body: Json): NormAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const toolCalls: NormToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : '',
      name: typeof block.name === 'string' ? block.name : '',
      argumentsJson: JSON.stringify(isRecord(block.input) ? block.input : {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: answerText(content),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    toolCalls,
  };
}

function anthropicAnswerFromSse(text: string): NormAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  const toolCalls = new Map<number, NormToolCall & { args: string }>();
  for (const frame of sseFrames(text)) {
    const data = parseJson(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message)) {
      const usage = isRecord(data.message.usage) ? data.message.usage : {};
      inputTokens = numberOrZero(usage.input_tokens);
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : toolCalls.size;
      toolCalls.set(index, {
        id: typeof data.content_block.id === 'string' ? data.content_block.id : '',
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
        argumentsJson: '{}',
        args: '',
      });
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const index = typeof data.index === 'number' ? data.index : -1;
        const entry = toolCalls.get(index);
        if (entry !== undefined) entry.args += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return {
    text: out,
    inputTokens,
    outputTokens,
    stop,
    toolCalls: [...toolCalls.values()].map((c) => ({ id: c.id, name: c.name, argumentsJson: c.args === '' ? '{}' : c.args })),
  };
}

function geminiAnswerFromParts(candidate: Json, usageSource: Json): NormAnswer {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let out = '';
  const toolCalls: NormToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') out += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: '',
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        argumentsJson: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  const usage = isRecord(usageSource.usageMetadata) ? usageSource.usageMetadata : {};
  return {
    text: out,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    toolCalls,
  };
}

function geminiAnswerFromJson(body: Json): NormAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  return geminiAnswerFromParts(candidate, body);
}

function geminiAnswerFromSse(text: string): NormAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'STOP';
  const toolCalls: NormToolCall[] = [];
  for (const frame of sseFrames(text)) {
    const data = parseJson(frame.data);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const piece = geminiAnswerFromParts(candidate, data);
      out += piece.text;
      toolCalls.push(...piece.toolCalls);
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: out, inputTokens, outputTokens, stop, toolCalls };
}

/** One provider answer body (JSON or SSE text) as a NormAnswer, in the provider's wire. */
export function parseAnswer(wire: ProviderWire, text: string, stream: boolean): NormAnswer {
  if (wire === 'openai') return stream ? openAIAnswerFromSse(text) : openAIAnswerFromJson(parseJson(text) ?? {});
  if (wire === 'anthropic') return stream ? anthropicAnswerFromSse(text) : anthropicAnswerFromJson(parseJson(text) ?? {});
  return stream ? geminiAnswerFromSse(text) : geminiAnswerFromJson(parseJson(text) ?? {});
}

// ---------------------------------------------------------------------------
// Rendering a NormAnswer in the client's wire.

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a text the way the simulated providers do: fixed-size pieces, one empty piece for ''. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

export function renderOpenAIJson(answer: NormAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id === '' ? 'call_agg' : c.id,
      type: 'function',
      function: { name: c.name, arguments: c.argumentsJson },
    }));
  }
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: mapStop(answer.stop, 'openai'), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

export function renderOpenAISse(answer: NormAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (answer.text !== '') {
    for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  }
  answer.toolCalls.forEach((call, index) => {
    frames.push(
      sseEvent(
        null,
        chunk(
          {
            tool_calls: [
              {
                index,
                id: call.id === '' ? 'call_agg' : call.id,
                type: 'function',
                function: { name: call.name, arguments: call.argumentsJson },
              },
            ],
          },
          null,
        ),
      ),
    );
  });
  frames.push(sseEvent(null, chunk({}, mapStop(answer.stop, 'openai'))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

export function renderAnthropicJson(answer: NormAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id === '' ? 'toolu_agg' : call.id, name: call.name, input: argumentsOf(call.argumentsJson) });
  }
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: mapStop(answer.stop, 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

export function renderAnthropicSse(answer: NormAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    if (answer.text !== '') {
      for (const piece of splitText(answer.text)) {
        frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
      }
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id === '' ? 'toolu_agg' : call.id, name: call.name, input: {} },
      }),
    );
    if (call.argumentsJson !== '' && call.argumentsJson !== '{}') {
      frames.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.argumentsJson } }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: mapStop(answer.stop, 'anthropic'), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** Render a normalized answer for one client: JSON text or the SSE byte stream, as the client asked. */
export function renderAnswer(client: ClientWire, answer: NormAnswer, model: string, stream: boolean): { contentType: string; body: string } {
  if (client === 'openai') {
    return stream
      ? { contentType: 'text/event-stream', body: renderOpenAISse(answer, model) }
      : { contentType: 'application/json', body: JSON.stringify(renderOpenAIJson(answer, model)) };
  }
  return stream
    ? { contentType: 'text/event-stream', body: renderAnthropicSse(answer, model) }
    : { contentType: 'application/json', body: JSON.stringify(renderAnthropicJson(answer, model)) };
}
