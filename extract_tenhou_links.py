import csv
from pathlib import Path

base_dir = Path(__file__).parent
csv_file = base_dir / "dataset/links/csv/☆孫燕姿☆ -- 天凤用户日志查询 - nodocchi.moe.csv"
output_file = base_dir / "dataset/links/tenhou.txt"

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)

    print(f"CSV表头: {header}")
    print(f"列索引 - 规则: 3, 牌谱: 4")

    tenhou_links = []
    total_rows = 0
    matched_rows = 0

    for row in reader:
        total_rows += 1
        if len(row) > 4:
            rule = row[3].strip()
            if rule == "四鳳南喰赤－":
                matched_rows += 1
                link = row[4].strip()
                if link.startswith('http'):
                    tenhou_links.append(link)

    print(f"\n统计信息:")
    print(f"总行数（不含表头）: {total_rows}")
    print(f"匹配'四鳳南喰赤－'的行数: {matched_rows}")
    print(f"提取的天凤链接数: {len(tenhou_links)}")

    with open(output_file, 'w', encoding='utf-8') as out:
        for link in tenhou_links:
            out.write(link + '\n')

    print(f"\n已将 {len(tenhou_links)} 个天凤链接写入: {output_file}")
