import { notFound } from "next/navigation";
import Link from "next/link";
import { getConceptBySlug, concepts } from "@/data/concepts";
import { getProblemsByCategory } from "@/data/problems";
import { DifficultyBadge } from "@/components/DifficultyBadge";
import { CodeBlock } from "@/components/CodeBlock";
import { Metadata } from "next";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return Object.keys(concepts).map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const concept = getConceptBySlug(slug);

  if (!concept) {
    return { title: "Concept Not Found" };
  }

  return {
    title: `${concept.name} | Pareto Problem Set`,
    description: `Learn the ${concept.name} pattern for coding interviews. ${concept.whenToUse[0]}`,
  };
}

export default async function ConceptPage({ params }: PageProps) {
  const { slug } = await params;
  const concept = getConceptBySlug(slug);

  if (!concept) {
    notFound();
  }

  const categoryProblems = getProblemsByCategory(slug);

  return (
    <main>
      <Link
        href="/"
        className="inline-flex items-center gap-1 text-base text-foreground/60 hover:text-accent transition-colors mb-6"
      >
        ← Back to All Problems
      </Link>

      <header className="mb-8">
        <h1 className="text-4xl font-bold text-foreground">{concept.name}</h1>
        <p className="mt-2 text-lg text-foreground/60">
          {categoryProblems.length} problems in this category
        </p>
      </header>

      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-3 text-foreground">
          When to Use
        </h2>
        <ul className="space-y-3">
          {concept.whenToUse.map((item, i) => (
            <li key={i} className="flex gap-3 text-foreground/80 text-base">
              <span className="text-accent">•</span>
              {item}
            </li>
          ))}
        </ul>
      </section>

      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-3 text-foreground">
          Common Patterns
        </h2>
        <ul className="space-y-3">
          {concept.commonPatterns.map((pattern, i) => (
            <li key={i} className="flex gap-3 text-foreground/80 text-base">
              <span className="text-difficulty-medium">→</span>
              {pattern}
            </li>
          ))}
        </ul>
      </section>

      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-3 text-foreground">
          Key Insights
        </h2>
        <ul className="space-y-3">
          {concept.keyInsights.map((insight, i) => (
            <li key={i} className="flex gap-3 text-foreground/80 text-base">
              <span className="text-difficulty-easy">✓</span>
              {insight}
            </li>
          ))}
        </ul>
      </section>

      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-3 text-foreground">
          Code Template
        </h2>
        <CodeBlock code={concept.codeTemplate} />
      </section>

      <section className="mb-8 p-5 bg-card-bg border border-card-border rounded-lg">
        <h2 className="text-base font-semibold mb-2 text-foreground/60 uppercase tracking-wide">
          Time Complexity Notes
        </h2>
        <p className="text-foreground/80 text-base">{concept.timeComplexityNotes}</p>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-4 text-foreground">
          Problems in This Category
        </h2>
        <div className="space-y-3">
          {categoryProblems.map((problem) => (
            <Link
              key={problem.id}
              href={`/problems/${problem.slug}`}
              className="flex items-center justify-between p-4 bg-card-bg border border-card-border rounded-lg hover:border-accent transition-colors group"
            >
              <div className="flex items-center gap-3">
                <span className="text-base text-foreground/50 font-mono w-7">
                  {String(problem.id).padStart(2, "0")}
                </span>
                <span className="text-foreground group-hover:text-accent transition-colors text-base">
                  {problem.name}
                </span>
              </div>
              <DifficultyBadge difficulty={problem.difficulty} />
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}
