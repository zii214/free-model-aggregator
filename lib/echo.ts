// The three wire parsers, mirrored from the env's own: a provider answers with an echo computed from
// the request AS IT PARSED IT, and the client compares that echo with its own parse of the body it
// sent. Routing a request across wires is therefore only safe when the provider's parse of the
// upstream body equals the client's parse of the client body on every echo field — which is what the
// guard in wire.ts checks, field by field, before any call is made.
//
// An echo holds: whether a system prompt was present and its sha, the last user message's text sha,
// the tool names in order, the max_tokens, the stop sequences, the last image's sha, and whether JSON
// mode was asked for. The shas are the first 8 hex of sha256: of the text, or of an image's bytes.

import { createHash } from 'node:crypto';
import { isRecord, type Json } from './brief.ts';

export interface Echo {
  system_present: boolean;
  system_sha8: string | null;
  user_sha8: string;
  tools: string[];
  max_tokens: number | null;
  stops: string[];
  image_sha8: string | null;
  json_mode: boolean;
}

/** What one wire's parser makes of a request body: its echo and the texts the usage is counted from. */
export interface Parsed {
  echo: Echo;
  systemText: string;
  userText: string;
  /** The stop token this wire's answers carry when they finish in family 0. */
  stop: string;
  stream: boolean;
  includeUsage: boolean;
  model: string | null;
}

export function sha8Text(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 8);
}

export function sha8Bytes(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex').slice(0, 8);
}

/** The text of a content value: itself when a string, else its text blocks joined with newlines. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** An image reference's sha: a data URL hashes its bytes, any other URL hashes the URL's text. */
export function imageShaOfUrl(url: unknown): string | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:[^,]*;base64,(.*)$/s.exec(url);
  if (data !== null) return sha8Bytes(Buffer.from(data[1], 'base64'));
  return sha8Text(url);
}

interface EchoParts {
  systemText: string;
  userText: string;
  tools: string[];
  maxTokens: number | null;
  stops: string[];
  imageSha8: string | null;
  jsonMode: boolean;
}

function echoOf(parts: EchoParts): Echo {
  return {
    system_present: parts.systemText !== '',
    system_sha8: parts.systemText === '' ? null : sha8Text(parts.systemText),
    user_sha8: sha8Text(parts.userText),
    tools: parts.tools,
    max_tokens: parts.maxTokens,
    stops: parts.stops,
    image_sha8: parts.imageSha8,
    json_mode: parts.jsonMode,
  };
}

/** The OpenAI wire's parse: system messages join, the LAST user message's text and last image count. */
export function openAIParsed(body: Json): Parsed {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'image_url' && isRecord(block.image_url)) imageSha8 = imageShaOfUrl(block.image_url.url) ?? imageSha8;
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
  const systemText = systemParts.join('\n');
  return {
    echo: echoOf({
      systemText,
      userText,
      tools,
      maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
      stops: stringList(body.stop),
      imageSha8,
      jsonMode: format === 'json_object' || format === 'json_schema',
    }),
    systemText,
    userText,
    stop: 'stop',
    stream: body.stream === true,
    includeUsage: isRecord(body.stream_options) && body.stream_options.include_usage === true,
    model: typeof body.model === 'string' ? body.model : null,
  };
}

/** The Anthropic wire's parse: `system` is its own field; JSON mode does not exist on this wire. */
export function anthropicParsed(body: Json): Parsed {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          imageSha8 = sha8Bytes(Buffer.from(block.source.data, 'base64'));
        } else if (block.source.type === 'url') {
          imageSha8 = imageShaOfUrl(block.source.url) ?? imageSha8;
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
  }
  return {
    echo: echoOf({
      systemText,
      userText,
      tools,
      maxTokens: numberOrNull(body.max_tokens),
      stops: stringList(body.stop_sequences),
      imageSha8,
      jsonMode: false,
    }),
    systemText,
    userText,
    stop: 'end_turn',
    stream: body.stream === true,
    includeUsage: false,
    model: typeof body.model === 'string' ? body.model : null,
  };
}

function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const texts: string[] = [];
  for (const part of parts) if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  return texts.join('\n');
}

/** The Gemini wire's parse: the model and the stream flag come from the call's path, not its body. */
export function geminiParsed(body: Json, model: string, stream: boolean): Parsed {
  const systemText = isRecord(body.systemInstruction) ? geminiPartsText(body.systemInstruction.parts) : '';
  const contents = Array.isArray(body.contents) ? body.contents.filter(isRecord) : [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const content of contents) {
    if (content.role === 'model') continue;
    userText = geminiPartsText(content.parts);
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (!isRecord(part)) continue;
        if (isRecord(part.inlineData) && typeof part.inlineData.data === 'string') {
          imageSha8 = sha8Bytes(Buffer.from(part.inlineData.data, 'base64'));
        } else if (isRecord(part.fileData) && typeof part.fileData.fileUri === 'string') {
          imageSha8 = imageShaOfUrl(part.fileData.fileUri) ?? imageSha8;
        }
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
  return {
    echo: echoOf({
      systemText,
      userText,
      tools,
      maxTokens: numberOrNull(config.maxOutputTokens),
      stops: stringList(config.stopSequences),
      imageSha8,
      jsonMode: config.responseMimeType === 'application/json',
    }),
    systemText,
    userText,
    stop: 'STOP',
    stream,
    includeUsage: false,
    model,
  };
}

/** The echo the client holds its answer to: its own wire's parse of the body it sent. */
export function clientEcho(api: 'openai' | 'anthropic', body: Json): Echo {
  return api === 'openai' ? openAIParsed(body).echo : anthropicParsed(body).echo;
}

/** The echo one wire's parser makes of an upstream body (gemini's model rides in the path). */
export function upstreamEcho(wire: 'openai' | 'anthropic' | 'gemini', body: Json, model: string, stream: boolean): Echo {
  if (wire === 'openai') return openAIParsed(body).echo;
  if (wire === 'anthropic') return anthropicParsed(body).echo;
  return geminiParsed(body, model, stream).echo;
}

function sameStrings(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

/** Whether two echoes agree on every field the client compares. */
export function echoEquals(a: Echo, b: Echo): boolean {
  return (
    a.system_present === b.system_present &&
    a.system_sha8 === b.system_sha8 &&
    a.user_sha8 === b.user_sha8 &&
    sameStrings(a.tools, b.tools) &&
    a.max_tokens === b.max_tokens &&
    sameStrings(a.stops, b.stops) &&
    a.image_sha8 === b.image_sha8 &&
    a.json_mode === b.json_mode
  );
}
