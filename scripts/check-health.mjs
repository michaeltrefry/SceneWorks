import process from "node:process";

const apiBaseUrl = process.env.SCENEWORKS_API_BASE_URL ?? "http://localhost:8000";

const response = await fetch(`${apiBaseUrl}/api/v1/health`);
if (!response.ok) {
  throw new Error(`Health check failed with HTTP ${response.status}`);
}

const health = await response.json();
if (health.status !== "ok" || health.service !== "sceneworks-api") {
  throw new Error(`Unexpected health payload: ${JSON.stringify(health)}`);
}

console.log(`SceneWorks API health ok at ${apiBaseUrl}/api/v1/health`);
