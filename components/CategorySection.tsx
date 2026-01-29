import Link from "next/link";
import { ProblemCard } from "./ProblemCard";
import { Problem } from "@/data/problems";

interface CategorySectionProps {
  name: string;
  slug: string;
  problems: Problem[];
}

export function CategorySection({
  name,
  slug,
  problems,
}: CategorySectionProps) {
  return (
    <section className="mb-10">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold text-foreground">{name}</h2>
        <Link
          href={`/concepts/${slug}`}
          className="text-base text-accent hover:text-accent-hover transition-colors"
        >
          View Concept →
        </Link>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {problems.map((problem) => (
          <ProblemCard
            key={problem.id}
            id={problem.id}
            slug={problem.slug}
            name={problem.name}
            difficulty={problem.difficulty}
          />
        ))}
      </div>
    </section>
  );
}
