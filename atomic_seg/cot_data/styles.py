"""Normal-success reasoning styles."""

STYLE_SPECS = (
    {
        "name": "evidence_first",
        "brief": (
            "Lead with the most diagnostic camera or state observations, then "
            "connect them causally to the next movement."
        ),
    },
    {
        "name": "temporal_contrast",
        "brief": (
            "Organize the explanation as previous-boundary versus current-state "
            "evidence; at an initial boundary explain the initial situation."
        ),
    },
    {
        "name": "goal_decomposition",
        "brief": (
            "Identify the immediate unresolved subgoal or prerequisite and "
            "explain why the requested atomic movement addresses it next."
        ),
    },
    {
        "name": "alternative_elimination",
        "brief": (
            "Discuss a meaningful alternative when the evidence supports one; "
            "do not invent alternatives to satisfy a style label."
        ),
    },
    {
        "name": "uncertainty_calibrated",
        "brief": (
            "Separate direct observations from uncertain physical inference and "
            "justify the next movement; describe contact when visually supported."
        ),
    },
)
