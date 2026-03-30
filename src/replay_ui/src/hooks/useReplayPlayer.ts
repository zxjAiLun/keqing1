// src/replay_ui/src/hooks/useReplayPlayer.ts
import { useState, useCallback, useRef, useEffect } from 'react';
import type { ReplayData } from '../types/replay';
import type { DecisionLogEntry } from '../types/replay';

export type PlaybackSpeed = 0.5 | 1 | 2 | 4;

export interface ReplayPlayerState {
  currentStep: number;
  isPlaying: boolean;
  speed: PlaybackSpeed;
  totalSteps: number;
  currentEntry: DecisionLogEntry | null;
  currentKyoku: number;
  totalKyoku: number;
}

export function useReplayPlayer(data: ReplayData | null) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<PlaybackSpeed>(1);
  const intervalRef = useRef<number | null>(null);

  const totalSteps = data?.log.length ?? 0;
  const totalKyoku = data?.kyoku_order.length ?? 0;

  const currentEntry: DecisionLogEntry | null =
    data && currentStep >= 0 && currentStep < data.log.length
      ? data.log[currentStep]
      : null;

  // 计算当前步所属的小局索引
  const currentKyoku = (() => {
    if (!data || !currentEntry) return 0;
    const key = currentEntry.kyoku_key;
    return data.kyoku_order.findIndex(
      k => k.bakaze === key.bakaze && k.kyoku === key.kyoku && k.honba === key.honba
    );
  })();

  // 播放定时器
  useEffect(() => {
    if (isPlaying && totalSteps > 0) {
      const interval = 800 / speed;
      intervalRef.current = window.setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= totalSteps - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, interval);
    }
    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isPlaying, speed, totalSteps]);

  const play = useCallback(() => {
    if (currentStep >= totalSteps - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(true);
  }, [currentStep, totalSteps]);

  const pause = useCallback(() => setIsPlaying(false), []);

  const togglePlay = useCallback(() => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }, [isPlaying, play, pause]);

  const stepForward = useCallback(() => {
    setCurrentStep(prev => Math.min(prev + 1, totalSteps - 1));
  }, [totalSteps]);

  const stepBackward = useCallback(() => {
    setCurrentStep(prev => Math.max(prev - 1, 0));
  }, []);

  const goToStart = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  const goToEnd = useCallback(() => {
    setCurrentStep(totalSteps - 1);
    setIsPlaying(false);
  }, [totalSteps]);

  const goToStep = useCallback((step: number) => {
    setCurrentStep(Math.max(0, Math.min(step, totalSteps - 1)));
  }, [totalSteps]);

  const goToKyoku = useCallback((kyokuIdx: number) => {
    if (!data) return;
    const key = data.kyoku_order[kyokuIdx];
    const firstStepIdx = data.log.findIndex(
      e => e.kyoku_key.bakaze === key.bakaze &&
           e.kyoku_key.kyoku === key.kyoku &&
           e.kyoku_key.honba === key.honba
    );
    if (firstStepIdx >= 0) {
      setCurrentStep(firstStepIdx);
      setIsPlaying(false);
    }
  }, [data]);

  const changeSpeed = useCallback((newSpeed: PlaybackSpeed) => {
    setSpeed(newSpeed);
  }, []);

  return {
    currentStep,
    isPlaying,
    speed,
    totalSteps,
    currentEntry,
    currentKyoku,
    totalKyoku,
    play,
    pause,
    togglePlay,
    stepForward,
    stepBackward,
    goToStart,
    goToEnd,
    goToStep,
    goToKyoku,
    changeSpeed,
  };
}
