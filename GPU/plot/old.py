import re
import pandas as pd

text = """
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 0.8% is 47780820/600000000
          total time: 11.826ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 0.8% is 47780820/600000000
          Total time: 59.9101ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 0.8% is 47780820/600000000
          Total time: 25.2631ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 0.8% is 47780820/600000000
          Total time: 12.288ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 0.8% is 47780820/600000000
          Total time: 18.8926ms
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 0.16% is 9581780/600000000
          total time: 11.1268ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 0.16% is 9581780/600000000
          Total time: 58.2268ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 0.16% is 9581780/600000000
          Total time: 24.4463ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 0.16% is 9581780/600000000
          Total time: 12.1743ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 0.16% is 9581780/600000000
          Total time: 18.9491ms
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 0.02% is 1259400/600000000
          total time: 10.8525ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 0.02% is 1259400/600000000
          Total time: 57.3092ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 0.02% is 1259400/600000000
          Total time: 24.15ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 0.02% is 1259400/600000000
          Total time: 11.9828ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 0.02% is 1259400/600000000
          Total time: 18.7098ms
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 3.428% is 206316430/600000000
          total time: 13.7206ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 3.428% is 206316430/600000000
          Total time: 125.49ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 3.428% is 206316430/600000000
          Total time: 28.0463ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 3.428% is 206316430/600000000
          Total time: 15.6815ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 3.428% is 206316430/600000000
          Total time: 20.436ms
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 0.13712% is 8183040/600000000
          total time: 10.0975ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 0.13712% is 8183040/600000000
          Total time: 96.6656ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 0.13712% is 8183040/600000000
          Total time: 24.3815ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 0.13712% is 8183040/600000000
          Total time: 9.71878ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 0.13712% is 8183040/600000000
          Total time: 18.8663ms
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 0.0054848% is 320540/600000000
          total time: 8.49101ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 0.0054848% is 320540/600000000
          Total time: 89.002ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 0.0054848% is 320540/600000000
          Total time: 24.051ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 0.0054848% is 320540/600000000
          Total time: 8.91699ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 0.0054848% is 320540/600000000
          Total time: 18.8242ms
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 7.68e-05% is 4050/600000000
          total time: 4.65174ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 7.68e-05% is 4050/600000000
          Total time: 25.9809ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 7.68e-05% is 4050/600000000
          Total time: 23.9862ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 7.68e-05% is 4050/600000000
          Total time: 3.84ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 7.68e-05% is 4050/600000000
          Total time: 18.7863ms
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 1.6% is 96363980/600000000
          total time: 13.6684ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 1.6% is 96363980/600000000
          Total time: 58.9928ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 1.6% is 96363980/600000000
          Total time: 36.1681ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 1.6% is 96363980/600000000
          Total time: 14.3067ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 1.6% is 96363980/600000000
          Total time: 31.4982ms
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 0.4576% is 27141630/600000000
          total time: 12.25ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 0.4576% is 27141630/600000000
          Total time: 56.7751ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 0.4576% is 27141630/600000000
          Total time: 34.263ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 0.4576% is 27141630/600000000
          Total time: 12.1682ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 0.4576% is 27141630/600000000
          Total time: 28.544ms
>>> Generated data 600000000 lines in lineorder table.
>>> Start test GPU OLAPcore using row-wise model
          Result count of total selection rate 0.009152% is 549310/600000000
          total time: 9.00506ms
>>> Start test GPU OLAPcore using Column-wise model and dynamic vector
          Result count of total selection rate 0.009152% is 549310/600000000
          Total time: 39.2357ms
>>> Start test GPU OLAPcore using Column-wise model and static vector
          Result count of total selection rate 0.009152% is 549310/600000000
          Total time: 34.1176ms
>>> Start test GPU OLAPcore using Vector-wise model and dynamic vector
          Result count of total selection rate 0.009152% is 549310/600000000
          Total time: 8.87091ms
>>> Start test GPU OLAPcore using Vector-wise model and static vector
          Result count of total selection rate 0.009152% is 549310/600000000
          Total time: 27.9542ms
"""

# 模型名称标准化规则
def normalize_model(desc: str) -> str:
    desc = desc.lower()
    if "row-wise" in desc:
        return "rowwise"
    if "column-wise" in desc and "dynamic" in desc:
        return "columnwise_dynamic_vector"
    if "column-wise" in desc and "static" in desc:
        return "columnwise_static_vector"
    if "vector-wise" in desc and "dynamic" in desc:
        return "vectorwise_dynamic_vector"
    if "vector-wise" in desc and "static" in desc:
        return "vectorwise_static_vector"
    return "unknown"

# 预定义的 TestCase 映射顺序，9 组数据
test_cases = ['2.1', '2.2', '2.3', '3.1', '3.2', '3.3', '3.4', '4.1', '4.2', '4.3']

lines = text.strip().splitlines()
data = []
case_index = -1
result_sum = 0

for line in lines:
    if "Generated data" in line:
        case_index += 1
        result_sum = None
    elif "Result count" in line:
        m = re.search(r'(\d+)/\d+', line)
        if m:
            result_sum = float(m.group(1))
    elif "using" in line:
        model = normalize_model(line)
    elif "time" in line.lower() and "ms" in line.lower():
        m = re.search(r'([\d.]+)ms', line)
        if m:
            time = float(m.group(1))
            if result_sum is not None:
                testcase = test_cases[case_index] if case_index < len(test_cases) else f"unknown-{case_index}"
                data.append([model, time, result_sum, testcase])

# 输出为 DataFrame
df = pd.DataFrame(data, columns=["Method", "Time", "TotalSum", "TestCase"])
df["code"] = "old"
# print(df.to_csv(index=False))
df.to_csv("SSB_benchmark_merged_old.csv", index=False)