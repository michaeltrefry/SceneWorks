import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import react from "eslint-plugin-react";
import globals from "globals";

// Flat ESLint config (sc-4197 / F-WEB-9): the web app had no lint step, so whole
// classes of trivially-detectable bugs (undefined globals, unused imports) only
// surfaced at runtime — two such bugs were found in the 2026-06-09 review. This
// enables the high-signal rules:
//   - no-undef        → would have caught F-WEB-1 (modelLoraFamilies) and F-WEB-5
//   - no-unused-vars  → flags dead imports
//   - react-hooks/rules-of-hooks (error) and exhaustive-deps (warn; the codebase
//     carries intentional eslint-disable-next-line exhaustive-deps comments).
export default [
  {
    ignores: ["dist/**", "node_modules/**", "coverage/**"],
  },
  js.configs.recommended,
  {
    files: ["src/**/*.{js,jsx}"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
      globals: {
        ...globals.browser,
      },
    },
    plugins: {
      "react-hooks": reactHooks,
      react,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-hooks/exhaustive-deps": "warn",
      // Without these, no-unused-vars treats JSX-only-used identifiers as unused:
      // jsx-uses-vars covers components rendered as <Foo/>; jsx-uses-react covers the
      // `React` import that the test files need (vitest uses the classic JSX runtime,
      // unlike the app build's automatic runtime).
      "react/jsx-uses-vars": "error",
      "react/jsx-uses-react": "error",
      "no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
    },
  },
  {
    // Test files run under Vitest (jsdom) with node + vitest globals.
    files: ["src/**/*.test.{js,jsx}", "src/**/*.spec.{js,jsx}"],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
        ...globals.vitest,
      },
    },
  },
];
