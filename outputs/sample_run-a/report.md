# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100_sample.json
- Mode: mock
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.62 | 0.68 | 0.06 |
| Avg attempts | 1 | 1.73 | 0.73 |
| Avg token estimate | 1437.08 | 2488.87 | 1051.79 |
| Avg latency (ms) | 887.5 | 2112.38 | 1224.88 |

## Failure modes
```json
{
  "react": {
    "wrong_final_answer": 38,
    "none": 62
  },
  "reflexion": {
    "wrong_final_answer": 32,
    "none": 68
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
