import { notFound } from "next/navigation";
import Link from "next/link";
import { getProblemBySlug, getAdjacentProblems, problems } from "@/data/problems";
import { DifficultyBadge } from "@/components/DifficultyBadge";
import { CodeBlock } from "@/components/CodeBlock";
import { Metadata } from "next";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return problems.map((problem) => ({
    slug: problem.slug,
  }));
}

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const problem = getProblemBySlug(slug);

  if (!problem) {
    return { title: "Problem Not Found" };
  }

  return {
    title: `${problem.name} | Pareto Problem Set`,
    description: `${problem.difficulty} - ${problem.category}. ${problem.explanation[0]}`,
  };
}

export default async function ProblemPage({ params }: PageProps) {
  const { slug } = await params;
  const problem = getProblemBySlug(slug);

  if (!problem) {
    notFound();
  }

  const { prev, next } = getAdjacentProblems(slug);

  return (
    <main>
      <Link
        href="/"
        className="inline-flex items-center gap-1 text-base text-foreground/60 hover:text-accent transition-colors mb-6"
      >
        ← Back to {problem.category}
      </Link>

      <header className="mb-8">
        <div className="flex items-start justify-between gap-4 mb-3">
          <h1 className="text-3xl font-bold text-foreground">
            <span className="text-foreground/50 font-mono mr-2">
              {String(problem.id).padStart(2, "0")}.
            </span>
            {problem.name}
          </h1>
          <DifficultyBadge difficulty={problem.difficulty} />
        </div>

        <div className="flex flex-wrap gap-4 text-base">
          <a
            href={problem.leetcodeUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent hover:text-accent-hover transition-colors"
          >
            LeetCode ↗
          </a>
          <a
            href={problem.neetcodeUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent hover:text-accent-hover transition-colors"
          >
            NeetCode ↗
          </a>
        </div>
      </header>

      <section className="mb-8">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xl font-semibold text-foreground">Solution</h2>
          <Link
            href={`/concepts/${problem.categorySlug}`}
            className="text-base text-accent hover:text-accent-hover transition-colors"
          >
            View Pattern →
          </Link>
        </div>
        <CodeBlock code={problem.solution} />
        <div className="mt-2 flex gap-4 text-sm text-foreground/50">
          <span>Time: <span className="font-mono">{problem.timeComplexity}</span></span>
          <span>Space: <span className="font-mono">{problem.spaceComplexity}</span></span>
        </div>
      </section>

      {problem.alternativeSolutions && problem.alternativeSolutions.length > 0 && (
        <section className="mb-8">
          <h2 className="text-xl font-semibold mb-4 text-foreground">
            Alternative Solutions
          </h2>
          <div className="space-y-6">
            {problem.alternativeSolutions.map((alt, i) => (
              <div key={i}>
                <h3 className="text-base font-medium text-foreground/80 mb-2">
                  {alt.title}
                </h3>
                <CodeBlock code={alt.code} />
                <div className="mt-2 flex gap-4 text-sm text-foreground/50">
                  <span>Time: <span className="font-mono">{alt.timeComplexity}</span></span>
                  <span>Space: <span className="font-mono">{alt.spaceComplexity}</span></span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-3 text-foreground">
          Explanation
        </h2>
        <ol className="space-y-3">
          {problem.explanation.map((step, i) => (
            <li
              key={i}
              className="flex gap-3 text-foreground/80 leading-relaxed text-base"
            >
              <span className="text-foreground/40 font-mono">
                {i + 1}.
              </span>
              {step}
            </li>
          ))}
        </ol>
      </section>

      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-3 text-foreground">
          Key Points
        </h2>
        <ul className="space-y-3">
          {problem.keyPoints.map((point, i) => (
            <li key={i} className="flex gap-3 text-foreground/80 text-base">
              <span className="text-accent">•</span>
              {point}
            </li>
          ))}
        </ul>
      </section>

      {/* Navigation */}
      <nav className="flex items-center justify-between pt-8 border-t border-card-border">
        {prev ? (
          <Link
            href={`/problems/${prev.slug}`}
            className="group flex flex-col items-start"
          >
            <span className="text-sm text-foreground/50 mb-1">← Previous</span>
            <span className="text-base text-foreground group-hover:text-accent transition-colors">
              {prev.name}
            </span>
            {prev.category !== problem.category && (
              <span className="text-sm text-foreground/40">{prev.category}</span>
            )}
          </Link>
        ) : (
          <div />
        )}

        {next ? (
          <Link
            href={`/problems/${next.slug}`}
            className="group flex flex-col items-end text-right"
          >
            <span className="text-sm text-foreground/50 mb-1">Next →</span>
            <span className="text-base text-foreground group-hover:text-accent transition-colors">
              {next.name}
            </span>
            {next.category !== problem.category && (
              <span className="text-sm text-foreground/40">{next.category}</span>
            )}
          </Link>
        ) : (
          <div />
        )}
      </nav>
    </main>
  );
}
