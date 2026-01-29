import Link from "next/link";
import { DifficultyBadge } from "./DifficultyBadge";

interface ProblemCardProps {
  id: number;
  slug: string;
  name: string;
  difficulty: "Easy" | "Medium" | "Hard";
}

export function ProblemCard({ id, slug, name, difficulty }: ProblemCardProps) {
  return (
    <Link
      href={`/problems/${slug}`}
      className="block p-4 bg-card-bg border border-card-border rounded-lg hover:border-accent transition-colors group"
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="text-base text-foreground/60 font-mono">
          {String(id).padStart(2, "0")}
        </span>
        <DifficultyBadge difficulty={difficulty} />
      </div>
      <h3 className="text-base font-medium text-foreground group-hover:text-accent transition-colors line-clamp-2">
        {name}
      </h3>
    </Link>
  );
}
