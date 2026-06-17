import { createContext, useContext } from "react";

// Narrow domain facade for the shared fullscreen preview. Screens still receive
// the legacy AppContext setPreviewAsset action during migration, while the shell
// overlay reads its own focused context.
export const PreviewContext = createContext(null);

export function usePreviewContext() {
  const value = useContext(PreviewContext);
  if (value === null) {
    throw new Error("usePreviewContext must be used within a <PreviewContext.Provider>");
  }
  return value;
}
