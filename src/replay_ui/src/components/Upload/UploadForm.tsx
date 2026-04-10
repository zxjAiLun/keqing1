// src/replay_ui/src/components/Upload/UploadForm.tsx
import { useState, useRef } from 'react';

interface UploadFormProps {
  onDataLoaded: (data: unknown) => void;
  onUploadStart?: () => void;
}

type InputType = 'tenhou_url' | 'mjai_json';

// 模型列表：从新到旧，第一个为「最新」
const MODEL_LIST: { value: string; label: string }[] = [
  { value: 'keqingv31', label: 'KeqingV3.1' },
  { value: 'keqingv3', label: 'KeqingV3' },
  { value: 'keqingv2', label: 'KeqingV2' },
  { value: 'keqingv1', label: 'KeqingV1' },
];
const LATEST_MODEL = MODEL_LIST[0].value;

const DEFAULT_TENHOU_URL = 'https://tenhou.net/3/?log=2021021820gm-00a9-0000-0b6677ca&tw=2';
const DEFAULT_MJAI_JSON = '[{"type":"start_game","names":["遊走","武田舞彩","九紋龍史進","Nemo"],"kyoku_first":0,"aka_flag":true},{"type":"start_kyoku","bakaze":"E","dora_marker":"9p","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","3m","6m","7m","1p","3p","6p","1s","1s","1s","2s","3s","5s"],["1m","3m","5m","6m","9p","2s","2s","2s","8s","9s","E","N","P"],["4m","5m","5pr","6p","8p","4s","6s","7s","7s","8s","9s","9s","S"],["2m","5mr","7m","8m","8m","2p","3p","8p","9p","8s","E","W","W"]]}]';

// ---------------------------------------------------------------------------
// 子组件：天凤链接输入
// ---------------------------------------------------------------------------
function TenhouUrlInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="https://tenhou.net/3/?log=...&tw=2"
        style={{
          width: '100%',
          padding: '10px 12px',
          border: '1px solid var(--border)',
          borderRadius: 8,
          fontSize: 13,
          fontFamily: '"Menlo", "Consolas", monospace',
          color: 'var(--text-primary)',
          background: 'var(--card-bg)',
          outline: 'none',
          boxSizing: 'border-box',
        }}
      />
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
        视角将自动从链接的 <code>tw=</code> 参数解析，无需手动选择
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 子组件：mjai JSONL 输入（textarea + 文件上传）
// ---------------------------------------------------------------------------
function MjaiJsonInput({
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
        📁 点击或拖拽一个 .json / .jsonl 文件
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
        placeholder='[{"type":"start_game",...}] ← 默认已填写示例 mjson'
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
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>支持粘贴或上传单个 mjai / JSON 文件；需手动选择视角</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 主组件
// ---------------------------------------------------------------------------
export function UploadForm({ onDataLoaded, onUploadStart }: UploadFormProps) {
  const [inputType, setInputType] = useState<InputType>('tenhou_url');
  const [tenhouUrl, setTenhouUrl] = useState(DEFAULT_TENHOU_URL);
  const [mjaiText, setMjaiText]   = useState(DEFAULT_MJAI_JSON);
  const [files, setFiles]         = useState<File[]>([]);
  const [playerId, setPlayerId]   = useState<string>('auto');
  const [botModel, setBotModel]   = useState<string>(LATEST_MODEL);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [success, setSuccess]     = useState<string | null>(null);

  // 切换输入类型时重置视角默认值
  const switchInputType = (t: InputType) => {
    setInputType(t);
    setPlayerId(t === 'tenhou_url' ? 'auto' : '1');
    setError(null);
    setSuccess(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = inputType === 'tenhou_url' ? tenhouUrl.trim() : mjaiText.trim();
    if (!text && files.length === 0) {
      setError('请填写链接或上传文件');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);
    onUploadStart?.();

    try {
      const formData = new FormData();

      if (inputType === 'tenhou_url') {
        formData.append('json_text', text);
        formData.append('input_type', 'url');
      } else {
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
        formData.append('input_type', 'mjai');
        for (const f of files) formData.append('files', f);
      }

      if (playerId !== 'auto') formData.append('player_id', playerId);
      formData.append('bot_type', botModel);

      const res = await fetch('/api/replay', { method: 'POST', body: formData });
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

  return (
    <form onSubmit={handleSubmit}>
      {/* 输入类型 Tab */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button type="button" style={tabStyle(inputType === 'tenhou_url')} onClick={() => switchInputType('tenhou_url')}>🔗 天凤链接</button>
        <button type="button" style={tabStyle(inputType === 'mjai_json')}  onClick={() => switchInputType('mjai_json')}>📋 mjai JSONL</button>
      </div>

      {/* 输入区域 */}
      <div style={{ marginBottom: 14 }}>
        {inputType === 'tenhou_url'
          ? <TenhouUrlInput value={tenhouUrl} onChange={setTenhouUrl} />
          : <MjaiJsonInput text={mjaiText} onTextChange={setMjaiText} files={files} onFilesChange={setFiles} />
        }
      </div>

      {/* 底部参数行 */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}>

        {/* 视角座位 */}
        <div style={{ flex: 1, minWidth: 160 }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6, display: 'block' }}>视角座位</label>
          <select value={playerId} onChange={e => setPlayerId(e.target.value)} style={selectStyle}>
            {inputType === 'tenhou_url' && <option value="auto">自动（来自链接 tw=）</option>}
            <option value="0">东家</option>
            <option value="1">南家</option>
            <option value="2">西家</option>
            <option value="3">北家</option>
          </select>
        </div>

        {/* 模型类型 */}
        <div style={{ flex: 1, minWidth: 130 }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6, display: 'block' }}>模型类型</label>
          <select value={botModel} onChange={e => setBotModel(e.target.value)} style={selectStyle}>
            <option value={LATEST_MODEL}>最新（{MODEL_LIST[0].label}）</option>
            {MODEL_LIST.slice(1).map(m => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
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
          ) : '▶ 运行 Review'}
        </button>
      </div>

      {error   && <div style={{ marginTop: 10, fontSize: 13, color: 'var(--error)' }}>{error}</div>}
      {success && <div style={{ marginTop: 10, fontSize: 13, color: 'var(--success)' }}>{success}</div>}
    </form>
  );
}
