import os
from typing import List

import numpy as np

# TODO: Test all functions
# TODO: Change the target transforms


def describe_pair_numerosity(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.seed(seed)

    first_obj_count = np.random.randint(0, max_num + 1)
    action = np.random.choice([-1, 0, 1])
    second_obj_count = first_obj_count + action
    while second_obj_count < 0 or second_obj_count > max_num:
        action = np.random.choice([-1, 0, 1])
        second_obj_count = first_obj_count + action
    background = np.random.choice(backgrounds)
    object_size = 46.2

    first_objects = [np.random.choice(animals + tools) for _ in range(first_obj_count)]
    second_objects = (
        first_objects
        if action == 0
        else first_objects[:-1]
        if action == -1
        else first_objects + [np.random.choice(animals + tools)]
    )

    return {
        "background": background,
        "objs": [
            {"size": object_size, "object": first_objects[i]}
            for i in range(first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-img.png",
    }, {
        "background": background,
        "objs": [
            {"size": object_size, "object": second_objects[i]}
            for i in range(second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-img.png",
    }


def describe_pair_numerosity_background_change(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.seed(seed)

    first_obj_count = np.random.randint(0, max_num + 1)
    action = np.random.choice([-1, 0, 1])
    second_obj_count = first_obj_count + action
    while second_obj_count < 0 or second_obj_count > max_num:
        action = np.random.choice([-1, 0, 1])
        second_obj_count = first_obj_count + action
    backgrounds = np.random.choice(backgrounds, replace=False, size=2)
    object_size = 46.2

    first_objects = [np.random.choice(animals + tools) for _ in range(first_obj_count)]
    second_objects = (
        first_objects
        if action == 0
        else first_objects[:-1]
        if action == -1
        else first_objects + [np.random.choice(animals + tools)]
    )

    return {
        "background": backgrounds[0],
        "objs": [
            {"size": object_size, "object": first_objects[i]}
            for i in range(first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-img.png",
    }, {
        "background": backgrounds[1],
        "objs": [
            {"size": object_size, "object": second_objects[i]}
            for i in range(second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-img.png",
    }


def describe_pair_numerosity_size_difference(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.seed(seed)

    first_obj_count = np.random.randint(0, max_num + 1)
    action = np.random.choice([-1, 0, 1])
    second_obj_count = first_obj_count + action
    while second_obj_count < 0 or second_obj_count > max_num:
        action = np.random.choice([-1, 0, 1])
        second_obj_count = first_obj_count + action

    background = np.random.choice(backgrounds)
    mean = 46.2
    dev = 0.4

    object_sizes_first = np.random.uniform(
        mean * (1 - dev), mean * (1 + dev), first_obj_count
    )
    if second_obj_count > first_obj_count:
        object_sizes_second = np.append(
            object_sizes_first, np.random.uniform(mean * (1 - dev), mean * (1 + dev), 1)
        )
    elif second_obj_count < first_obj_count:
        object_sizes_second = object_sizes_first[:-1]
    else:
        object_sizes_second = object_sizes_first

    first_objects = [np.random.choice(animals + tools) for _ in range(first_obj_count)]
    second_objects = (
        first_objects
        if action == 0
        else first_objects[:-1]
        if action == -1
        else first_objects + [np.random.choice(animals + tools)]
    )

    return {
        "background": background,
        "objs": [
            {"size": object_sizes_first[i], "object": first_objects[i]}
            for i in range(first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-img.png",
    }, {
        "background": background,
        "objs": [
            {"size": object_sizes_second[i], "object": second_objects[i]}
            for i in range(second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-img.png",
    }


def describe_pair_varying_size_fixed_image_fixed_between(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.seed(seed)

    b = 0.5
    max_size_allowed = 100
    min_size_allowed = 22
    std_dev_ratio = 0.2

    first_obj_count = np.random.randint(0, max_num + 1)
    action = np.random.choice([-1, 0, 1])
    second_obj_count = first_obj_count + action
    while second_obj_count < 0 or second_obj_count > max_num:
        action = np.random.choice([-1, 0, 1])
        second_obj_count = first_obj_count + action
    background = np.random.choice(backgrounds)

    avg = (max_size_allowed - min_size_allowed) * np.exp(
        -b * first_obj_count
    ) + min_size_allowed
    std_dev = avg * std_dev_ratio
    object_sizes_first = np.clip(
        np.random.normal(avg, std_dev, first_obj_count),
        min_size_allowed,
        max_size_allowed,
    )

    avg = (max_size_allowed - min_size_allowed) * np.exp(
        -b * second_obj_count
    ) + min_size_allowed
    std_dev = avg * std_dev_ratio
    object_sizes_second = np.clip(
        np.random.normal(avg, std_dev, second_obj_count),
        min_size_allowed,
        max_size_allowed,
    )

    object_type = np.random.choice(animals + tools)
    first_objects = [object_type for _ in range(first_obj_count)]
    second_objects = (
        first_objects
        if action == 0
        else first_objects[:-1]
        if action == -1
        else first_objects + [object_type]
    )

    return {
        "background": background,
        "objs": [
            {"size": object_sizes_first[i], "object": first_objects[i]}
            for i in range(first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-img.png",
    }, {
        "background": background,
        "objs": [
            {"size": object_sizes_second[i], "object": second_objects[i]}
            for i in range(second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-img.png",
    }

def describe_pair_varying_size(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
): # TODO: Change name. This is: only fixed between images but same in pair with all identites possible
    np.random.seed(seed)

    b = 0.5
    max_size_allowed = 100
    min_size_allowed = 22
    std_dev_ratio = 0.2

    first_obj_count = np.random.randint(0, max_num + 1)
    action = np.random.choice([-1, 0, 1])
    second_obj_count = first_obj_count + action
    while second_obj_count < 0 or second_obj_count > max_num:
        action = np.random.choice([-1, 0, 1])
        second_obj_count = first_obj_count + action
    background = np.random.choice(backgrounds)

    avg = (max_size_allowed - min_size_allowed) * np.exp(
        -b * first_obj_count
    ) + min_size_allowed
    std_dev = avg * std_dev_ratio
    object_sizes_first = np.clip(
        np.random.normal(avg, std_dev, first_obj_count),
        min_size_allowed,
        max_size_allowed,
    )

    avg = (max_size_allowed - min_size_allowed) * np.exp(
        -b * second_obj_count
    ) + min_size_allowed
    std_dev = avg * std_dev_ratio
    object_sizes_second = np.clip(
        np.random.normal(avg, std_dev, second_obj_count),
        min_size_allowed,
        max_size_allowed,
    )

    first_objects = [np.random.choice(animals + tools) for _ in range(first_obj_count)]
    second_objects = (
        first_objects
        if action == 0
        else first_objects[:-1]
        if action == -1
        else first_objects + [np.random.choice(animals + tools)]
    )

    return {
        "background": background,
        "objs": [
            {"size": object_sizes_first[i], "object": first_objects[i]}
            for i in range(first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-img.png",
    }, {
        "background": background,
        "objs": [
            {"size": object_sizes_second[i], "object": second_objects[i]}
            for i in range(second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-img.png",
    }


def describe_pair_varying_size_and_object( 
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
): # This is with unfixed between and within
    np.random.seed(seed)

    b = 0.5
    max_size_allowed = 100
    min_size_allowed = 22
    std_dev_ratio = 0.2

    first_obj_count = np.random.randint(0, max_num + 1)
    action = np.random.choice([-1, 0, 1])
    second_obj_count = first_obj_count + action
    while second_obj_count < 0 or second_obj_count > max_num:
        action = np.random.choice([-1, 0, 1])
        second_obj_count = first_obj_count + action
    background = np.random.choice(backgrounds)

    avg = (max_size_allowed - min_size_allowed) * np.exp(
        -b * first_obj_count
    ) + min_size_allowed
    std_dev = avg * std_dev_ratio
    object_sizes_first = np.clip(
        np.random.normal(avg, std_dev, first_obj_count),
        min_size_allowed,
        max_size_allowed,
    )

    avg = (max_size_allowed - min_size_allowed) * np.exp(
        -b * second_obj_count
    ) + min_size_allowed
    std_dev = avg * std_dev_ratio
    object_sizes_second = np.clip(
        np.random.normal(avg, std_dev, second_obj_count),
        min_size_allowed,
        max_size_allowed,
    )

    first_objects = [np.random.choice(animals + tools) for _ in range(first_obj_count)]
    second_objects = [
        np.random.choice(list(set(animals + tools) - set(first_objects)))
        for _ in range(second_obj_count)
    ]

    return {
        "background": background,
        "objs": [
            {"size": object_sizes_first[i], "object": first_objects[i]}
            for i in range(first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-img.png",
    }, {
        "background": background,
        "objs": [
            {"size": object_sizes_second[i], "object": second_objects[i]}
            for i in range(second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-img.png",
    }

def describe_pair_varying_size_and_object_and_background( 
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
): # This is with unfixed between and within
    np.random.seed(seed)

    b = 0.5
    max_size_allowed = 100
    min_size_allowed = 22
    std_dev_ratio = 0.2

    first_obj_count = np.random.randint(0, max_num + 1)
    action = np.random.choice([-1, 0, 1])
    second_obj_count = first_obj_count + action
    while second_obj_count < 0 or second_obj_count > max_num:
        action = np.random.choice([-1, 0, 1])
        second_obj_count = first_obj_count + action
    background_first = np.random.choice(backgrounds)
    background_second = np.random.choice(list(set(backgrounds) - set([background_first])))

    avg = (max_size_allowed - min_size_allowed) * np.exp(
        -b * first_obj_count
    ) + min_size_allowed
    std_dev = avg * std_dev_ratio
    object_sizes_first = np.clip(
        np.random.normal(avg, std_dev, first_obj_count),
        min_size_allowed,
        max_size_allowed,
    )

    avg = (max_size_allowed - min_size_allowed) * np.exp(
        -b * second_obj_count
    ) + min_size_allowed
    std_dev = avg * std_dev_ratio
    object_sizes_second = np.clip(
        np.random.normal(avg, std_dev, second_obj_count),
        min_size_allowed,
        max_size_allowed,
    )

    first_objects = [np.random.choice(animals + tools) for _ in range(first_obj_count)]
    second_objects = [
        np.random.choice(list(set(animals + tools) - set(first_objects)))
        for _ in range(second_obj_count)
    ]

    return {
        "background": background_first,
        "objs": [
            {"size": object_sizes_first[i], "object": first_objects[i]}
            for i in range(first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-img.png",
    }, {
        "background": background_second,
        "objs": [
            {"size": object_sizes_second[i], "object": second_objects[i]}
            for i in range(second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-img.png",
    }

def describe_pair_numerosity_object_change(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.seed(seed)

    first_obj_count = np.random.randint(0, max_num + 1)
    action = np.random.choice([-1, 0, 1])
    second_obj_count = first_obj_count + action
    while second_obj_count < 0 or second_obj_count > max_num:
        action = np.random.choice([-1, 0, 1])
        second_obj_count = first_obj_count + action
    background = np.random.choice(backgrounds)
    object_size = 46.2

    first_objects = [np.random.choice(animals + tools) for _ in range(first_obj_count)]
    second_objects = [
        np.random.choice(list(set(animals + tools) - set(first_objects)))
        for _ in range(second_obj_count)
    ]

    return {
        "background": background,
        "objs": [
            {"size": object_size, "object": first_objects[i]}
            for i in range(first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-img.png",
    }, {
        "background": background,
        "objs": [
            {"size": object_size, "object": second_objects[i]}
            for i in range(second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-img.png",
    }


def describe_pair_category_differentiation(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.count(seed)
    numerosity = np.random.randint(0, max_num)
    background = np.random.choice(backgrounds)
    object_size = 49.2

    first = np.random.choice(["Tools", "Animals"])
    second = np.random.choice(["Tools", "Animals"])
    label = (
        "Different" if first != second else "Animals" if first == "Animals" else "Tools"
    )
    label = "None" if numerosity == 0 else label

    return {
        "background": background,
        "objs": [
            {
                "size": object_size,
                "object": np.random.choice(tools if first == "Tools" else animals),
            }
            for _ in range(numerosity)
        ],
        "name_suffix": f"-{numerosity}-{label}-img.png",
    }, {
        "background": background,
        "objs": [
            {
                "size": object_size,
                "object": np.random.choice(tools if second == "Tools" else animals),
            }
            for _ in range(numerosity)
        ],
        "name_suffix": f"-{numerosity}-{label}-img.png",
    }


def describe_pair_even_odd(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.count(seed)
    object_size = 49.2
    background = np.random.choice(backgrounds)

    first_obj_count = np.random.randint(0, max_num)
    second_obj_count = first_obj_count + np.random.choice([-1, 0, 1])
    while second_obj_count < 0 or second_obj_count > max_num:
        second_obj_count = first_obj_count + np.random.choice([-1, 0, 1])

    def even(x):
        return x % 2 == 0

    label = (
        "even"
        if even(first_obj_count) and even(second_obj_count) == 0
        else "odd"
        if not even(first_obj_count) and not even(second_obj_count)
        else "different"
    )

    return {
        "background": background,
        "objs": [
            {
                "size": object_size,
                "object": np.random.choice(animals + tools),
            }
            for _ in range(first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-{label}-img.png",
    }, {
        "background": background,
        "objs": [
            {
                "size": object_size,
                "object": np.random.choice(animals + tools),
            }
            for _ in range(second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-{label}-img.png",
    }


def describe_pair_fixed_object(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.count(seed)
    object_size = 49.2
    background = np.random.choice(backgrounds)

    numerosity = np.random.randint(0, max_num)
    fixed_numerosity = np.random.randint(0, numerosity)

    fixed_objs = np.random.choice(animals + tools, size=fixed_numerosity, replace=True)
    fixed_objects = [
        {"size": object_size, "object": fixed_objs[i]} for i in range(fixed_numerosity)
    ]

    first_residual_objs = np.random.choice(
        list(set(animals + tools) - set(fixed_objs)),
        size=numerosity - fixed_numerosity,
        replace=True,
    )
    first_residual_objects = [
        {
            "size": object_size,
            "object": first_residual_objs[i],
        }
        for i in range(numerosity - fixed_numerosity)
    ]
    second_residual_objects = [
        {
            "size": object_size,
            "object": np.random.choice(
                list(set(animals + tools) - set(fixed_objs) - set(first_residual_objs))
            ),
        }
        for _ in range(numerosity - fixed_numerosity)
    ]

    return {
        "background": background,
        "objs": first_residual_objects + fixed_objects,
        "name_suffix": f"-{numerosity}-{fixed_numerosity}-img.png",
    }, {
        "background": background,
        "objs": second_residual_objects,
        "name_suffix": f"-{numerosity}-{fixed_numerosity}-img.png",
    }


def describe_pair_object_differentiation(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.count(seed)
    object_size = 49.2
    background = np.random.choice(backgrounds)

    numerosity = np.random.randint(0, max_num)
    object1 = "CameraLeica"
    object2 = "Horse"

    label1 = np.random.choice([object1, object2])
    label2 = np.random.choice([object1, object2])
    if label1 != label2:
        label1 = "Different"
        label2 = "Different"

    return {
        "background": background,
        "objs": [{"size": object_size, "object": label1} for _ in range(numerosity)],
        "name_suffix": f"-{numerosity}-{label1}-img.png",
    }, {
        "background": background,
        "objs": [{"size": object_size, "object": label2} for _ in range(numerosity)],
        "name_suffix": f"-{numerosity}-{label2}-img.png",
    }


def describe_pair_target_number(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.count(seed)
    object_size = 49.2
    background = np.random.choice(backgrounds)

    numerosity = np.random.randint(0, max_num)
    target_object = "CameraLeica"
    first_target_count = np.random.randint(0, numerosity)
    second_target_count = np.random.randint(0, numerosity)

    return {
        "background": background,
        "objs": [
            {
                "size": object_size,
                "object": np.random.choice(
                    list(set(animals + tools) - set([target_object]))
                ),
            }
            for _ in range()
        ]
        + [
            {
                "size": object_size,
                "object": target_object,
            }
            for _ in range(second_target_count)
        ],
        "name_suffix": f"-{numerosity}-{first_target_count}-img.png",
    }, {
        "background": background,
        "objs": [
            {
                "size": object_size,
                "object": np.random.choice(
                    list(set(animals + tools) - set([target_object]))
                ),
            }
            for _ in range(numerosity - second_target_count)
        ]
        + [
            {
                "size": object_size,
                "object": target_object,
            }
            for _ in range(second_target_count)
        ],
        "name_suffix": f"-{numerosity}-{second_target_count}-img.png",
    }


def describe_pair_target_object(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.count(seed)
    object_size = 49.2
    target_object = "CameraLeica"
    background = np.random.choice(backgrounds)
    numerosity = np.random.randint(0, max_num)
    label = np.random.choice(["first", "second", "none"])

    return {
        "background": background,
        "objs": [
            {
                "size": object_size,
                "object": np.random.choice(animals + tools),
            }
            for _ in range(numerosity - 1 if label == "first" else numerosity)
        ]
        + [
            (
                [
                    {
                        "size": object_size,
                        "object": target_object,
                    }
                ]
                if label == "first"
                else []
            )
        ],
        "name_suffix": f"-{numerosity}-{label}-img.png",
    }, {
        "background": background,
        "objs": [
            {
                "size": object_size,
                "object": np.random.choice(animals + tools),
            }
            for _ in range(numerosity - 1 if label == "second" else numerosity)
        ]
        + [
            (
                [
                    {
                        "size": object_size,
                        "object": target_object,
                    }
                ]
                if label == "first"
                else []
            )
        ],
        "name_suffix": f"-{numerosity}-{label}-img.png",
    }


def describe_pair_fill_objects_numerosity(
    seed: int,
    max_num: int,
    backgrounds: List[str],
    animals: List[str],
    tools: List[str],
):
    np.random.seed(seed)
    object_size = 46.2
    fill_numerosity = max_num
    filler_object = "CameraLeica"
    background = np.random.choice(backgrounds)

    first_obj_count = np.random.randint(0, max_num)

    action = np.random.choice([-1, 0, 1])
    second_obj_count = first_obj_count + action
    while second_obj_count < 0 or second_obj_count > max_num:
        action = np.random.choice([-1, 0, 1])
        second_obj_count = first_obj_count

    valid_objects = [obj for obj in animals + tools if obj != filler_object]

    first_objects: str = [
        np.random.choice(valid_objects) for _ in range(first_obj_count)
    ]
    second_objects = (
        first_objects
        if action == 0
        else first_objects[:-1]
        if action == -1
        else first_objects + [np.random.choice(valid_objects)]
    )

    return {
        "background": background,
        "objs": [
            {
                "size": object_size,
                "object": first_objects[i],
            }
            for i in range(first_obj_count)
        ]
        + [
            {"size": object_size, "object": filler_object}
            for _ in range(fill_numerosity - first_obj_count)
        ],
        "name_suffix": f"-{first_obj_count}-img.png",
    }, {
        "background": background,
        "objs": [
            {"size": object_size, "object": second_objects[i]}
            for i in range(second_obj_count)
        ]
        + [
            {"size": object_size, "object": filler_object}
            for _ in range(fill_numerosity - second_obj_count)
        ],
        "name_suffix": f"-{second_obj_count}-img.png",
    }


# ********* Helpers **********


def _list_backgrounds(render_dir: str) -> List[str]:
    backgrounds_dir = os.path.join(render_dir, "Backgrounds")
    return [f for f in os.listdir(backgrounds_dir) if not f.startswith(".")]


def _list_animals(render_dir: str) -> List[str]:
    animals_dir = os.path.join(render_dir, "Objects", "Animals")
    return [
        f
        for f in os.listdir(animals_dir)
        if os.path.isdir(os.path.join(animals_dir, f))
    ]


def _list_tools(render_dir: str) -> List[str]:
    tools_dir = os.path.join(render_dir, "Objects", "Tools")
    return [
        f for f in os.listdir(tools_dir) if os.path.isdir(os.path.join(tools_dir, f))
    ]
