# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100_sample.json
- Mode: mock
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.39 | 0.42 | 0.03 |
| Avg attempts | 1 | 2.21 | 1.21 |
| Avg token estimate | 1436.39 | 3156.4 | 1720.01 |
| Avg latency (ms) | 1189.22 | 2587.23 | 1398.01 |

## Failure modes
```json
{
  "react": {
    "wrong_final_answer": 61,
    "none": 39
  },
  "reflexion": {
    "wrong_final_answer": 58,
    "none": 42
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
