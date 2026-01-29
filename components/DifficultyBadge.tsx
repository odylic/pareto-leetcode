interface DifficultyBadgeProps {
  difficulty: "Easy" | "Medium" | "Hard";
}

export function DifficultyBadge({ difficulty }: DifficultyBadgeProps) {
  const colors = {
    Easy: "bg-difficulty-easy/20 text-difficulty-easy",
    Medium: "bg-difficulty-medium/20 text-difficulty-medium",
    Hard: "bg-difficulty-hard/20 text-difficulty-hard",
  };

  return (
    <span
      className={`px-2.5 py-1 rounded text-sm font-medium ${colors[difficulty]}`}
    >
      {difficulty}
    </span>
  );
}
