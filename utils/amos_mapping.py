"""
AMOS22 dataset class mapping
"""

# AMOS22の15臓器クラスマッピング
AMOS22_CLASS_MAP = {
    1: "spleen",
    2: "right kidney",
    3: "left kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior vena cava",
    10: "pancreas",
    11: "right adrenal gland",
    12: "left adrenal gland",
    13: "duodenum",
    14: "bladder",
    15: "prostate/uterus",
}

# 臓器名からラベルIDへの逆引きマップ
AMOS22_NAME_TO_ID = {v: k for k, v in AMOS22_CLASS_MAP.items()}


def get_organ_label_id(organ_name):
    """
    臓器名からラベルIDを取得

    Args:
        organ_name: 臓器名（例: "liver", "spleen"）

    Returns:
        ラベルID (1-15)
    """
    return AMOS22_NAME_TO_ID.get(organ_name)


def get_organ_name(label_id):
    """
    ラベルIDから臓器名を取得

    Args:
        label_id: ラベルID (1-15)

    Returns:
        臓器名
    """
    return AMOS22_CLASS_MAP.get(label_id)


def generate_fixed_label_for_continual_learning(task_organ_ids, total_organs=15, is_multi_label=True, only_new_organ=True):
    """
    継続学習用のラベルマッピングを生成

    Args:
        task_organ_ids: 現在のタスクで学習する臓器のIDリスト（例: [1, 2]）
        total_organs: 全臓器数
        is_multi_label: Trueの場合、各臓器に異なるラベルを割り当て
                        Falseの場合、全ての選択臓器にラベル1を割り当て
        only_new_organ: Trueの場合、最新の臓器のみラベル付け、それ以外は背景(0)
                       Falseの場合、全ての累積臓器にラベル付け

    Returns:
        orig_labels: 元のラベルリスト [1, 2, 3, ..., 15]
        target_labels: マッピング後のラベルリスト
    """
    orig_labels = list(range(1, total_organs + 1))
    target_labels = [0] * total_organs  # 初期値は全て0（背景）

    if only_new_organ:
        # 最新の臓器のみをラベル付け（継続学習の各タスクで新しい臓器のみ学習）
        # Task 0: [1] -> organ 1 = label 1, others = 0
        # Task 1: [1,2] -> organ 2 = label 2, others (including 1) = 0
        # Task 2: [1,2,3] -> organ 3 = label 3, others (including 1,2) = 0
        if len(task_organ_ids) > 0:
            new_organ_id = task_organ_ids[-1]  # 最新の臓器
            if 1 <= new_organ_id <= total_organs:
                target_labels[new_organ_id - 1] = new_organ_id
    else:
        # 累積的にラベル付け（全ての過去の臓器も含む）
        if is_multi_label:
            # 各臓器に異なるラベルを割り当て
            for idx, organ_id in enumerate(task_organ_ids):
                if 1 <= organ_id <= total_organs:
                    target_labels[organ_id - 1] = organ_id
        else:
            # 全ての選択臓器にラベル1を割り当て
            for organ_id in task_organ_ids:
                if 1 <= organ_id <= total_organs:
                    target_labels[organ_id - 1] = 1

    return orig_labels, target_labels


def print_label_mapping(orig_labels, target_labels):
    """
    ラベルマッピングを表示

    Args:
        orig_labels: 元のラベルリスト
        target_labels: マッピング後のラベルリスト
    """
    print("Label Mapping:")
    print(f"  Original labels: {orig_labels}")
    print(f"  Target labels:   {target_labels}")
    print("\nMapping details:")
    for orig, target in zip(orig_labels, target_labels):
        organ_name = AMOS22_CLASS_MAP.get(orig, "unknown")
        if target > 0:
            print(f"  {orig} ({organ_name}) -> {target}")
        else:
            print(f"  {orig} ({organ_name}) -> 0 (background)")
