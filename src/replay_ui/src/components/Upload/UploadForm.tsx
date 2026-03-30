// src/replay_ui/src/components/Upload/UploadForm.tsx
import { useState, useRef } from 'react';

interface UploadFormProps {
  onDataLoaded: (data: unknown) => void;
  onUploadStart?: () => void;
}

type InputType = 'tenhou_url' | 'mjai_json';

const DEFAULT_TENHOU_URL = 'https://tenhou.net/3/?log=2021021820gm-00a9-0000-0b6677ca&tw=2';
const DEFAULT_MJAI_JSON = '[{"type":"start_game","names":["遊走","武田舞彩","九紋龍史進","Nemo"],"kyoku_first":0,"aka_flag":true},{"type":"start_kyoku","bakaze":"E","dora_marker":"9p","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","3m","6m","7m","1p","3p","6p","1s","1s","1s","2s","3s","5s"],["1m","3m","5m","6m","9p","2s","2s","2s","8s","9s","E","N","P"],["4m","5m","5pr","6p","8p","4s","6s","7s","7s","8s","9s","9s","S"],["2m","5mr","7m","8m","8m","2p","3p","8p","9p","8s","E","W","W"]]}]';

export function UploadForm({ onDataLoaded, onUploadStart }: UploadFormProps) {
  const [inputType, setInputType] = useState<InputType>('mjai_json');
  const [text, setText] = useState(DEFAULT_MJAI_JSON);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() && files.length === 0) {
      setError('请上传文件或粘贴内容');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);
    onUploadStart?.();

    try {
      const formData = new FormData();
      let jsonText = text.trim();

      if (inputType === 'tenhou_url') {
        formData.append('json_text', jsonText);
        formData.append('input_type', 'url');
      } else {
        // mjai JSONL
        const lines = jsonText.split('\n').filter(l => l.trim());
        let events;
        try {
          if (lines.length > 1 || !jsonText.includes('"type"')) {
            events = lines.map(l => JSON.parse(l));
          } else {
            events = JSON.parse(jsonText);
          }
        } catch {
          throw new Error('JSON 格式错误，请检查输入');
        }
        formData.append('json_text', JSON.stringify(events));
        formData.append('input_type', 'mjai');
      }

      formData.append('player_id', '0');
      formData.append('bot_type', 'keqingv1');

      for (const f of files) formData.append('files', f);

      const res = await fetch('/api/replay', {
        method: 'POST',
        body: formData,
      });

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

  return (
    <form onSubmit={handleSubmit}>
      {/* 输入类型 */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 14, flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', margin: 0, fontWeight: 400, fontSize: 13, color: '#374151' }}>
          <input
            type="radio"
            checked={inputType === 'tenhou_url'}
            onChange={() => { setInputType('tenhou_url'); setText(DEFAULT_TENHOU_URL); }}
            style={{ margin: 0, cursor: 'pointer' }}
          />
          天凤链接
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', margin: 0, fontWeight: 400, fontSize: 13, color: '#374151' }}>
          <input
            type="radio"
            checked={inputType === 'mjai_json'}
            onChange={() => { setInputType('mjai_json'); setText(DEFAULT_MJAI_JSON); }}
            style={{ margin: 0, cursor: 'pointer' }}
          />
          mjai JSONL
        </label>
      </div>

      {/* 上传区 */}
      <div
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: '2px dashed #d1d5db',
          borderRadius: 8,
          padding: 20,
          textAlign: 'center',
          cursor: 'pointer',
          color: '#9ca3af',
          fontSize: 13,
          marginBottom: 10,
          transition: 'border-color 0.2s, background 0.2s',
        }}
        onDragOver={e => { e.preventDefault(); }}
        onDrop={e => {
          e.preventDefault();
          const dropped = Array.from(e.dataTransfer.files);
          setFiles(prev => [...prev, ...dropped.filter(f => !prev.some(p => p.name === f.name))]);
        }}
      >
        📁 点击或拖拽文件到此处上传
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,.jsonl,.txt"
          multiple
          style={{ display: 'none' }}
          onChange={e => {
            const chosen = Array.from(e.target.files || []);
            setFiles(prev => [...prev, ...chosen.filter(f => !prev.some(p => p.name === f.name))]);
          }}
        />
      </div>

      {/* 文件列表 */}
      {files.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
          {files.map(f => (
            <div key={f.name} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: '#f3f4f6', border: '1px solid #e5e7eb', borderRadius: 4, padding: '2px 8px', fontSize: 12 }}>
              📄 {f.name}
              <span
                onClick={() => setFiles(prev => prev.filter(p => p.name !== f.name))}
                style={{ cursor: 'pointer', color: '#9ca3af', fontWeight: 'bold' }}
              >
                ×
              </span>
            </div>
          ))}
        </div>
      )}

      {/* 文本框 */}
      <textarea
        ref={textareaRef}
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder={
          inputType === 'tenhou_url'
            ? 'https://tenhou.net/3/?log=2021021820gm-00a9-0000-0b6677ca&tw=2'
            : '[{"type":"start_game","names":[...],...}] ← 默认已填写示例 mjson'
        }
        style={{
          width: '100%',
          height: 110,
          padding: '10px 12px',
          border: '1px solid #d1d5db',
          borderRadius: 6,
          fontSize: 13,
          fontFamily: '"Menlo", "Consolas", monospace',
          resize: 'vertical',
          color: '#1f2937',
          background: '#fff',
          outline: 'none',
        }}
      />
      <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>
        {inputType === 'tenhou_url' ? '自动从链接的 tw= 参数获取视角，无需手动选择' : 'mjai JSONL 格式；需手动选择视角'}
      </div>

      {/* 提交按钮 */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end', marginTop: 14 }}>
        <div style={{ flex: 1, minWidth: 160 }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 6, display: 'block' }}>视角座位</label>
          <select
            style={{
              width: '100%',
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 14,
              background: '#fff',
              color: '#1f2937',
            }}
          >
            <option value="auto">自动（来自链接 tw=）</option>
            <option value="0">东家</option>
            <option value="1" selected>南家</option>
            <option value="2">西家</option>
            <option value="3">北家</option>
          </select>
        </div>
        <div style={{ flex: 1, minWidth: 130 }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 6, display: 'block' }}>模型类型</label>
          <select
            style={{
              width: '100%',
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 14,
              background: '#fff',
              color: '#1f2937',
            }}
          >
            <option value="keqingv1" selected>KeqingV1</option>
            <option value="keqingv2">KeqingV2</option>
            <option value="v5">V5Model</option>
          </select>
        </div>
        <div style={{ flex: 2, minWidth: 200 }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 6, display: 'block' }}>模型路径</label>
          <input
            type="text"
            defaultValue="artifacts/models/keqingv1/latest.pth"
            style={{
              width: '100%',
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 14,
              background: '#fff',
              color: '#1f2937',
            }}
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          style={{
            height: 40,
            padding: '0 28px',
            background: '#3498db',
            color: '#fff',
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
          }}
        >
          {loading ? (
            <>
              <span style={{ display: 'inline-block', width: 16, height: 16, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
              处理中…
            </>
          ) : '▶ 运行 Review'}
        </button>
      </div>

      {/* 错误 */}
      {error && (
        <div style={{ marginTop: 10, fontSize: 13, color: '#dc2626' }}>{error}</div>
      )}
      {success && (
        <div style={{ marginTop: 10, fontSize: 13, color: '#16a34a' }}>{success}</div>
      )}
    </form>
  );
}
