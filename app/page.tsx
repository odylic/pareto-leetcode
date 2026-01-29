import { CategorySection } from "@/components/CategorySection";
import { problems, categories } from "@/data/problems";

export default function Home() {
  return (
    <main>
      <header className="mb-12 text-center">
        <h1 className="text-5xl font-bold mb-4 text-foreground">
          The Pareto Problem Set
        </h1>
        <p className="text-xl text-foreground/70 max-w-2xl mx-auto">
          49 problems to pass 90% of technical interviews. Master these patterns
          and you&apos;ll be ready for anything.
        </p>
        <div className="mt-4 flex justify-center gap-4 text-base text-foreground/50">
          <span>49 Problems</span>
          <span>·</span>
          <span>9 Categories</span>
          <span>·</span>
          <span>Python Solutions</span>
        </div>
      </header>

      {categories.map((category) => {
        const categoryProblems = problems.filter(
          (p) => p.categorySlug === category.slug
        );
        return (
          <CategorySection
            key={category.slug}
            name={category.name}
            slug={category.slug}
            problems={categoryProblems}
          />
        );
      })}

    </main>
  );
}
