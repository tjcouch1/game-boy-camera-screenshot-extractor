import { useState, useEffect } from "react";

export type ClipboardPermissionState = "granted" | "denied" | "prompt" | "unknown";

export function useClipboardPermission(): ClipboardPermissionState {
  const [state, setState] = useState<ClipboardPermissionState>("unknown");

  useEffect(() => {
    let active = true;
    let permissionStatus: PermissionStatus | null = null;

    const checkPermission = async () => {
      try {
        const status = await navigator.permissions.query({ name: "clipboard-read" as PermissionName });
        if (!active) return;
        permissionStatus = status;
        setState(status.state as ClipboardPermissionState);
        
        status.onchange = () => {
          if (active) setState(status.state as ClipboardPermissionState);
        };
      } catch (err) {
        // Some browsers (like Firefox) don't support querying clipboard-read
        if (active) setState("unknown");
      }
    };

    checkPermission();

    return () => {
      active = false;
      if (permissionStatus) {
        permissionStatus.onchange = null;
      }
    };
  }, []);

  return state;
}
