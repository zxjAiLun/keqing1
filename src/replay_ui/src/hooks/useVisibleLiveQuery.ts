// src/replay_ui/src/hooks/useVisibleLiveQuery.ts
// 三页（天梯/账号/模型）共享的"页面可见时实时刷新"查询生命周期。
//
// 语义：
// - 首次加载：loading=true，失败显示 error，data=null；
// - 静默轮询：页面 visible 时每 intervalMs 执行一次，hidden 时跳过，恢复 visible 立即执行；
// - 轮询失败：保留当前 data，不替换成整页错误，下一轮继续重试；
// - 轮询成功：原子替换 data，清除旧 error，从失败状态恢复；
// - queryKey 改变：abort 旧请求、清空旧实体数据、开始新实体首次加载；
// - 卸载：abort 请求、移除 interval 与 visibilitychange listener。
//
// 同一实例保证最多一个在途请求（inFlightRef + AbortController），
// 不使用 setInterval(async () => ...) 直接堆请求。
import { useCallback, useEffect, useRef, useState } from 'react';

export interface VisibleLiveQueryOptions<T> {
  /** false 时不加载、不轮询（如缺少 activeSeasonId）。 */
  enabled: boolean;
  /** 变化即清空旧数据并重新首次加载；旧请求被 abort。 */
  queryKey: string;
  load: (signal: AbortSignal) => Promise<T>;
  intervalMs?: number;
}

export interface VisibleLiveQueryResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  /** 静默轮询进行中（非首次加载）。 */
  refreshing: boolean;
}

export function useVisibleLiveQuery<T>(options: VisibleLiveQueryOptions<T>): VisibleLiveQueryResult<T> {
  const { enabled, queryKey, load, intervalMs = 30_000 } = options;
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadRef = useRef(load);
  const intervalMsRef = useRef(intervalMs);
  const inFlightRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const queryKeyRef = useRef(queryKey);
  const disposedRef = useRef(false);

  useEffect(() => {
    loadRef.current = load;
  }, [load]);

  useEffect(() => {
    intervalMsRef.current = intervalMs;
  }, [intervalMs]);

  const run = useCallback(async (mode: 'initial' | 'poll') => {
    if (inFlightRef.current) return;
    if (document.visibilityState !== 'visible') return;
    inFlightRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    if (mode === 'poll') setRefreshing(true);
    try {
      const payload = await loadRef.current(controller.signal);
      if (disposedRef.current || controller.signal.aborted) return;
      setData(payload);
      setError(null);
      setLoading(false);
    } catch (reason) {
      if (disposedRef.current || controller.signal.aborted) return;
      // AbortError 属于 queryKey 切换/卸载的正常取消，不作为用户错误
      if (reason instanceof DOMException && reason.name === 'AbortError') return;
      if (mode === 'initial') {
        setError(reason instanceof Error ? reason.message : String(reason));
        setLoading(false);
      }
      // 轮询失败：保留现有 data，等待下一轮重试
    } finally {
      inFlightRef.current = false;
      if (abortRef.current === controller) abortRef.current = null;
      if (mode === 'poll') setRefreshing(false);
    }
  }, []);

  // queryKey 变化：abort 旧请求 -> 清空旧数据 -> 首次加载
  useEffect(() => {
    if (!enabled) return;
    if (queryKey !== queryKeyRef.current) {
      abortRef.current?.abort();
      inFlightRef.current = false;
      queryKeyRef.current = queryKey;
      setData(null);
      setLoading(true);
      setError(null);
    }
    void run('initial');
  }, [enabled, queryKey, run]);

  // 可见性轮询：visible 每 intervalMs；hidden 跳过；恢复 visible 立即刷新
  useEffect(() => {
    if (!enabled) return;
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') void run('poll');
    }, intervalMsRef.current);
    const onVisibility = () => {
      if (document.visibilityState === 'visible') void run('poll');
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [enabled, run]);

  // 卸载：abort 在途请求并标记已释放
  useEffect(() => {
    disposedRef.current = false;
    return () => {
      disposedRef.current = true;
      abortRef.current?.abort();
    };
  }, []);

  return { data, loading, error, refreshing };
}
