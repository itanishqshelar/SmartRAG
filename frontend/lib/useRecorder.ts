import { useCallback, useEffect, useRef, useState } from "react";

/** Record microphone audio with MediaRecorder. `stop()` resolves with the recording. */
export function useRecorder() {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timer = useRef<ReturnType<typeof setInterval>>(undefined);

  const cleanup = useCallback(() => {
    clearInterval(timer.current);
    recorder.current?.stream.getTracks().forEach((t) => t.stop());
    recorder.current = null;
    setRecording(false);
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    chunks.current = [];
    mr.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    mr.start();
    recorder.current = mr;
    setSeconds(0);
    setRecording(true);
    timer.current = setInterval(() => setSeconds((s) => s + 1), 1000);
  }, []);

  const stop = useCallback(
    () =>
      new Promise<Blob | null>((resolve) => {
        const mr = recorder.current;
        if (!mr) return resolve(null);
        mr.onstop = () => {
          resolve(new Blob(chunks.current, { type: mr.mimeType || "audio/webm" }));
          cleanup();
        };
        mr.stop();
      }),
    [cleanup],
  );

  const cancel = useCallback(() => {
    if (recorder.current) recorder.current.onstop = null;
    recorder.current?.stop();
    cleanup();
  }, [cleanup]);

  return { recording, seconds, start, stop, cancel };
}
