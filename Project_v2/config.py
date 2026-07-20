num_steps = 120

algorithms = [
    "heuristic_no_delay",
    "heuristic_low_delay",
    "heuristic_high_delay", 
    "agentic"
]
# ]

colors = {
    "heuristic_no_delay": "blue",
    "heuristic_low_delay": "orange",
    "heuristic_high_delay": "red",
    "agentic": "green",
}

algo_delays = {
    "heuristic_no_delay": 0.0,
    "heuristic_low_delay": 0.5,
    "heuristic_high_delay": 10.0,
}

metric_names = ["total", "acceptance_rate"]

random_seed = 50

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "sk-or-v1-..."

verbosity_level = 1  # 0: no print, 1: few prints (important messages), 2: more prints (detailed)

network_args = {
    "num_nodes": 15,
    "link_prob": 0.5,
    "capacity_min": 20,
    "capacity_max": 80,
    "latency_min": 1,
    "latency_max": 10,
    "loss_min": 0.0,
    "loss_max": 0.02,
    "fail_p_min": 0.001,
    "fail_p_max": 0.01,
}

base_demand = 8.0

traffic_pattern_args = {
    "latency_requirement_min_ms": 4.0,
    "latency_requirement_max_ms": 8.0,
}

traffic_args = {
    "seasonal_amplitude": 0.0,
    "seasonal_frequency": 24.0,
    "noise_power": 0.0,
}


POLICY_REFRESH_STEPS = 10
LLM_TOP_K_AFFECTED_FLOWS = 10



#keys to compare models
ACTIVE_AGENT_MODEL = "local_llama"

MODEL_CONFIGS = {
    # Local model through Ollama.
    
    "local_llama": {
        "provider": "ollama",
        "model": "llama3.2",
        "temperature": 0.0,
        "keep_alive": "30m",
        "num_ctx": 2048,
    },


    "requesty_gpt": {
        "provider": "requesty",
        "model": "openai/gpt-5.4-nano",
        "temperature": 0.0,
        "max_tokens": 256,
    },
    "requesty_gemini": {
        "provider": "requesty",
        "model": "google/gemini-2.5-flash",
        "temperature": 0.0,
        "max_tokens": 1000,
    },
    "requesty_claude": {
        "provider": "requesty",
        "model": "anthropic/claude-sonnet-4-5",
        "temperature": 0.0,
        "max_tokens": 768,
    },
    "requesty_claude_haiku": {
        "provider": "requesty",
        "model": "anthropic/claude-haiku-4-5",
        "temperature": 0.0,
        "max_tokens": 512,
    },

   

    "requesty_groq_20b": {
        "provider": "requesty",
        "model": "groq/openai/gpt-oss-20b",
        "temperature": 0.0,
        "max_tokens": 200,
    },

    "requesty_groq_120b": {
        "provider": "requesty",
        "model": "groq/openai/gpt-oss-120b",
        "temperature": 0.0,
        "max_tokens": 200,
    },

    "requesty_deepseek": {
        "provider": "requesty",
        "model": "deepseek/deepseek-v4-flash",
        "temperature": 0.0,
        # DeepSeek may spend several hundred completion tokens on internal
        # reasoning before producing the visible JSON response.
        "max_tokens": 1200,
    },
}
