// src/replay_ui/src/components/Upload/UploadForm.tsx
import { useState, useRef } from 'react';
import type { BotType } from '../../types/bot';
import { GUI_BOT_CATALOG } from '../../utils/botCatalog';

interface UploadFormProps {
  onDataLoaded: (data: unknown) => void;
  onUploadStart?: () => void;
}

type InputType = 'tenhou_url' | 'mjai_json' | 'tenhou6_json';

const TENHOU6_JSON_MARKER = '#json=';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseTenhou6JsonPayload(payload: string): Record<string, unknown> {
  const trimmed = payload.trim();
  const candidates = [trimmed];
  try {
    const decoded = decodeURIComponent(trimmed);
    if (decoded !== trimmed) candidates.push(decoded);
  } catch {
    // Keep the raw fragment; malformed percent escapes will be reported as JSON errors below.
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (isRecord(parsed) && Array.isArray(parsed.log)) return parsed;
    } catch {
      // Try the next representation.
    }
  }
  throw new Error('tenhou6 链接中的 json 不是有效牌谱对象');
}

function parseTenhou6JsonLinks(text: string): Record<string, unknown> | null {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const payloads = lines
    .map((line) => {
      const idx = line.indexOf(TENHOU6_JSON_MARKER);
      return idx >= 0 ? line.slice(idx + TENHOU6_JSON_MARKER.length) : null;
    })
    .filter((payload): payload is string => Boolean(payload?.trim()));

  if (payloads.length === 0) return null;
  if (payloads.length !== lines.length) {
    throw new Error('检测到 tenhou6 json 链接时，请每行只放一个 tenhou.net/6/#json=... 链接');
  }

  const parsed = payloads.map(parseTenhou6JsonPayload);
  if (parsed.length === 1) return parsed[0];

  const [first, ...rest] = parsed;
  return {
    ...first,
    log: [first, ...rest].flatMap((item) => Array.isArray(item.log) ? item.log : []),
  };
}

function parseTenhou6TextInput(text: string): Record<string, unknown> {
  const fromLinks = parseTenhou6JsonLinks(text);
  if (fromLinks) return fromLinks;
  const parsed = JSON.parse(text);
  if (!isRecord(parsed) || !Array.isArray(parsed.log)) {
    throw new Error('tenhou6 JSON 需要是包含 log 数组的对象');
  }
  return parsed;
}

function isTenhou6JsonLinkText(text: string): boolean {
  return text.includes(TENHOU6_JSON_MARKER);
}

// ---------------------------------------------------------------------------
// 子组件：天凤链接输入
// ---------------------------------------------------------------------------
function TenhouUrlInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        style={{
          width: '100%',
          height: 74,
          padding: '10px 12px',
          border: '1px solid var(--border)',
          borderRadius: 8,
          fontSize: 13,
          fontFamily: '"Menlo", "Consolas", monospace',
          resize: 'vertical',
          color: 'var(--text-primary)',
          background: 'var(--card-bg)',
          outline: 'none',
          boxSizing: 'border-box',
        }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// 子组件：mjai JSONL 输入（textarea + 文件上传）
// ---------------------------------------------------------------------------
function JsonReplayInput({
  text,
  onTextChange,
  files,
  onFilesChange,
}: {
  text: string;
  onTextChange: (v: string) => void;
  files: File[];
  onFilesChange: (files: File[]) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addFiles = (incoming: File[]) => {
    const next = incoming.find(Boolean);
    onFilesChange(next ? [next] : []);
  };

  return (
    <div>
      {/* 拖拽上传区 */}
      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); addFiles(Array.from(e.dataTransfer.files)); }}
        style={{
          border: '2px dashed var(--border)',
          borderRadius: 8,
          padding: 16,
          textAlign: 'center',
          cursor: 'pointer',
          color: 'var(--text-muted)',
          fontSize: 13,
          marginBottom: 8,
          background: 'var(--page-bg)',
          transition: 'border-color 0.2s, background 0.2s',
        }}
      >
        选择文件
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,.jsonl,.txt"
          style={{ display: 'none' }}
          onChange={e => addFiles(Array.from(e.target.files || []))}
        />
      </div>

      {/* 文件列表 */}
      {files.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
          {files.map(f => (
            <div key={f.name} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: 'var(--page-bg)', border: '1px solid var(--border)', borderRadius: 999, padding: '4px 10px', fontSize: 12, color: 'var(--text-secondary)' }}>
              📄 {f.name}
              <span
                onClick={() => onFilesChange(files.filter(p => p.name !== f.name))}
                style={{ cursor: 'pointer', color: 'var(--text-muted)', fontWeight: 'bold' }}
              >×</span>
            </div>
          ))}
        </div>
      )}

      {/* 文本粘贴区 */}
      <textarea
        value={text}
        onChange={e => onTextChange(e.target.value)}
        style={{
          width: '100%',
          height: 110,
          padding: '10px 12px',
          border: '1px solid var(--border)',
          borderRadius: 8,
          fontSize: 13,
          fontFamily: '"Menlo", "Consolas", monospace',
          resize: 'vertical',
          color: 'var(--text-primary)',
          background: 'var(--card-bg)',
          outline: 'none',
          boxSizing: 'border-box',
        }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// 主组件
// ---------------------------------------------------------------------------
export function UploadForm({ onDataLoaded, onUploadStart }: UploadFormProps) {
  const [inputType, setInputType] = useState<InputType>('tenhou_url');
  const [tenhouUrl, setTenhouUrl] = useState('');
  const [mjaiText, setMjaiText]   = useState('');
  const [tenhou6Text, setTenhou6Text] = useState('');
  const [files, setFiles]         = useState<File[]>([]);
  const [playerId, setPlayerId]   = useState<string>('auto');
  const [selectedModels, setSelectedModels] = useState<BotType[]>(['ext_mortal', '70k', 'mortal']);
  const [nagaUrl, setNagaUrl] = useState('');
  const [mortalUrl, setMortalUrl] = useState('');
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [success, setSuccess]     = useState<string | null>(null);
  const tenhouUrlIsTenhou6Json = inputType === 'tenhou_url' && isTenhou6JsonLinkText(tenhouUrl);
  const playerIdValue = tenhouUrlIsTenhou6Json && playerId === 'auto' ? '0' : playerId;

  // 切换输入类型时重置视角默认值
  const switchInputType = (t: InputType) => {
    setInputType(t);
    setPlayerId(t === 'tenhou_url' ? 'auto' : '1');
    setFiles([]);
    setError(null);
    setSuccess(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = inputType === 'tenhou_url'
      ? tenhouUrl.trim()
      : inputType === 'tenhou6_json'
        ? tenhou6Text.trim()
        : mjaiText.trim();
    if (!text && files.length === 0) {
      setError('请填写链接或上传文件');
      return;
    }
    if (selectedModels.length === 0) {
      setError('至少选择一个 Mortal checkpoint');
      return;
    }
    for (const [label, value] of [['NAGA', nagaUrl], ['Mortal 4.1c', mortalUrl]]) {
      if (!value.trim()) continue;
      try {
        const parsed = new URL(value.trim());
        if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') throw new Error();
      } catch {
        setError(`${label} 链接格式错误`);
        return;
      }
    }

    setLoading(true);
    setError(null);
    setSuccess(null);
    onUploadStart?.();

    try {
      const formData = new FormData();

      if (inputType === 'tenhou_url') {
        const linkedTenhou6 = parseTenhou6JsonLinks(text);
        if (linkedTenhou6) {
          formData.append('json_text', JSON.stringify(linkedTenhou6));
          formData.append('input_type', 'tenhou6');
        } else {
          formData.append('json_text', text);
          formData.append('input_type', 'url');
        }
      } else if (inputType === 'tenhou6_json') {
        if (files.length === 0) {
          let data: Record<string, unknown>;
          try {
            data = parseTenhou6TextInput(text);
          } catch {
            throw new Error('tenhou6 JSON 格式错误，请检查输入');
          }
          formData.append('json_text', JSON.stringify(data));
        }
        formData.append('input_type', 'tenhou6');
        for (const f of files) formData.append('files', f);
      } else {
        if (files.length === 0) {
          const lines = text.split('\n').filter(l => l.trim());
          let events;
          try {
            if (lines.length > 1 || !text.includes('"type"')) {
              events = lines.map(l => JSON.parse(l));
            } else {
              events = JSON.parse(text);
            }
          } catch {
            throw new Error('JSON 格式错误，请检查输入');
          }
          formData.append('json_text', JSON.stringify(events));
        }
        formData.append('input_type', 'mjai');
        for (const f of files) formData.append('files', f);
      }

      if (playerIdValue !== 'auto') formData.append('player_id', playerIdValue);
      for (const model of selectedModels) {
        formData.append('model_types', model);
      }
      if (nagaUrl.trim()) formData.append('naga_url', nagaUrl.trim());
      if (mortalUrl.trim()) formData.append('mortal_url', mortalUrl.trim());

      const res = await fetch('/api/replay/multi-teacher', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `请求失败: ${res.status}`);
      }

      const data = await res.json();
      onDataLoaded(data);
      setSuccess(`回放加载成功！共 ${data.log?.length ?? 0} 步`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const tabStyle = (active: boolean): React.CSSProperties => ({
    padding: '7px 20px',
    fontSize: 13,
    fontWeight: active ? 600 : 400,
    color: active ? '#fff' : 'var(--text-secondary)',
    background: active ? 'var(--accent)' : 'transparent',
    border: '1px solid',
    borderColor: active ? 'var(--accent)' : 'var(--border)',
    borderRadius: 8,
    cursor: 'pointer',
    transition: 'all 0.15s',
  });

  const selectStyle: React.CSSProperties = {
    width: '100%',
    padding: '8px 12px',
    border: '1px solid var(--border)',
    borderRadius: 8,
    fontSize: 14,
    background: 'var(--card-bg)',
    color: 'var(--text-primary)',
  };

  const toggleModel = (model: BotType) => {
    setSelectedModels((current) =>
      current.includes(model)
        ? current.filter((item) => item !== model)
        : [...current, model],
    );
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* 输入类型 Tab */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button type="button" style={tabStyle(inputType === 'tenhou_url')} onClick={() => switchInputType('tenhou_url')}>🔗 天凤链接</button>
        <button type="button" style={tabStyle(inputType === 'tenhou6_json')} onClick={() => switchInputType('tenhou6_json')}>📦 tenhou6 JSON</button>
        <button type="button" style={tabStyle(inputType === 'mjai_json')}  onClick={() => switchInputType('mjai_json')}>📋 mjai JSONL</button>
      </div>

      {/* 输入区域 */}
      <div style={{ marginBottom: 14 }}>
        {inputType === 'tenhou_url' && <TenhouUrlInput value={tenhouUrl} onChange={setTenhouUrl} />}
        {inputType === 'tenhou6_json' && (
          <JsonReplayInput
            text={tenhou6Text}
            onTextChange={setTenhou6Text}
            files={files}
            onFilesChange={setFiles}
          />
        )}
        {inputType === 'mjai_json' && (
          <JsonReplayInput
            text={mjaiText}
            onTextChange={setMjaiText}
            files={files}
            onFilesChange={setFiles}
          />
        )}
      </div>

      {/* 外部 Review 链接（NAGA / Mortal），其 Q 值会与本地模型一同渲染 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 8, marginBottom: 14 }}>
        <label style={externalLinkLabelStyle}>
          <span>NAGA Review</span>
          <input type="url" value={nagaUrl} onChange={(event) => setNagaUrl(event.target.value)} style={externalLinkInputStyle} />
        </label>
        <label style={externalLinkLabelStyle}>
          <span>Mortal 4.1c Review</span>
          <input
            type="url"
            value={mortalUrl}
            onChange={(event) => setMortalUrl(event.target.value)}
            placeholder="https://mjai.ekyu.moe/.../?data=/report/xxxx.json"
            style={externalLinkInputStyle}
          />
        </label>
      </div>

      {/* 底部参数行 */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}>

        {/* 视角座位 */}
        <div style={{ flex: 1, minWidth: 160 }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6, display: 'block' }}>视角座位</label>
          <select value={playerIdValue} onChange={e => setPlayerId(e.target.value)} style={selectStyle}>
            {inputType === 'tenhou_url' && !tenhouUrlIsTenhou6Json && <option value="auto">自动（来自链接 tw=）</option>}
            <option value="0">东家</option>
            <option value="1">南家</option>
            <option value="2">西家</option>
            <option value="3">北家</option>
          </select>
        </div>

        {/* 模型类型 */}
        <div style={{ flex: '2 1 360px', minWidth: 260 }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6, display: 'block' }}>Mortal checkpoint 对比</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 6 }}>
            {GUI_BOT_CATALOG.map((bot) => {
              const active = selectedModels.includes(bot.value);
              return (
                <button
                  key={bot.value}
                  type="button"
                  onClick={() => toggleModel(bot.value)}
                  style={{
                    minHeight: 36,
                    padding: '7px 9px',
                    borderRadius: 7,
                    border: `1px solid ${active ? '#8e44ad' : 'var(--border)'}`,
                    background: active ? 'rgba(142,68,173,0.10)' : 'var(--card-bg)',
                    color: active ? '#8e44ad' : 'var(--text-primary)',
                    cursor: 'pointer',
                    textAlign: 'left',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
                    <span style={{ fontSize: 12, fontWeight: 800 }}>{bot.shortLabel}</span>
                    {active && <span style={{ fontSize: 10, color: '#8e44ad' }}>已选</span>}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* 提交按钮 */}
        <button
          type="submit"
          disabled={loading}
          style={{
            height: 40,
            padding: '0 28px',
            background: 'var(--btn-primary-bg)',
            color: 'var(--btn-primary-text)',
            border: 'none',
            borderRadius: 8,
            fontSize: 15,
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1,
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            whiteSpace: 'nowrap',
            alignSelf: 'flex-end',
          }}
        >
          {loading ? (
            <><span style={{ display: 'inline-block', width: 16, height: 16, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />处理中…</>
          ) : '▶ 运行多模型 Review'}
        </button>
      </div>

      {error   && <div style={{ marginTop: 10, fontSize: 13, color: 'var(--error)' }}>{error}</div>}
      {success && <div style={{ marginTop: 10, fontSize: 13, color: 'var(--success)' }}>{success}</div>}
    </form>
  );
}

const externalLinkLabelStyle: React.CSSProperties = {
  display: 'grid',
  gap: 5,
  color: 'var(--text-primary)',
  fontSize: 12,
  fontWeight: 700,
};

const externalLinkInputStyle: React.CSSProperties = {
  width: '100%',
  height: 34,
  border: '1px solid var(--border)',
  borderRadius: 6,
  padding: '0 9px',
  background: 'var(--card-bg)',
  color: 'var(--text-primary)',
  fontFamily: 'Menlo, Consolas, monospace',
  fontSize: 11,
  boxSizing: 'border-box',
};
