---
name: local-review
description: Pre-PR quality review - verify AC coverage, test quality, E2E preference, and test isolation.
---

# Local Review

Quality enforcement for pre-PR review. Use before creating a PR to catch issues early.

## Quick Start

```bash
# Start the workflow
kspec workflow start @local-review
kspec workflow next --input spec_ref="@spec-slug"
```

## When to Use

- Before creating a PR
- When spawning a review subagent
- After completing implementation, before shipping

## Workflow Overview

6-step quality gate with strict checks:

1. **Get Spec & ACs** - Read acceptance criteria
2. **AC Coverage** - Every AC must have test (MUST-FIX)
3. **Test Quality** - No fluff tests (MUST-FIX)
4. **Test Strategy** - Prefer E2E over unit
5. **Test Isolation** - Tests properly isolated (MUST-FIX)
6. **Report** - List all blocking issues

## Review Criteria

### 1. AC Coverage (MUST-FIX)

Every acceptance criterion MUST have at least one test that validates it.

**How to check:**
```bash
# Find AC annotations in tests
grep -r "// AC: @spec-ref" tests/

# Compare against spec ACs
kspec item get @spec-ref
```

**Annotation format:**
```typescript
// AC: @spec-ref ac-1
it('should validate input when given invalid data', () => {
  // Test implementation
});
```

Missing AC coverage is a **blocking issue**, not a suggestion.

### 2. Test Quality (MUST-FIX)

All tests must properly validate their intended purpose.

**Valid tests:**
- AC-specific tests that validate acceptance criteria
- Edge case tests that catch real bugs
- Integration tests that verify components work together

**Fluff tests to reject:**
- Tests that always pass regardless of implementation
- Tests that only verify implementation details
- Tests that mock everything and verify nothing

**Litmus test:** Would this test fail if the feature breaks?

### 3. Test Strategy (Advisory)

Prefer end-to-end tests over unit tests.

**Good:** Test the feature as a user would interact with it
```typescript
it('should complete the workflow end-to-end', async () => {
  const result = await runFeature(testInput);
  expect(result.status).toBe('success');
  expect(result.output).toContain('expected-value');
});
```

**Less good:** Only unit testing internal functions
```typescript
it('should format output', () => {
  const formatted = formatOutput(mockData);
  expect(formatted).toBe('...');
});
```

Unit tests are okay for complex logic, but E2E proves the feature works.

### 4. Test Isolation (MUST-FIX)

All tests MUST run in isolated environments (temp directories, mocked services, etc.).

**Why this matters:**
- Prevents side effects between tests
- Prevents data corruption in real project state
- Ensures tests are reproducible

**Correct pattern:**
```typescript
let tempDir: string;

beforeEach(async () => {
  tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'test-'));
  await setupFixtures(tempDir);
});

afterEach(async () => {
  await fs.rm(tempDir, { recursive: true, force: true });
});
```

**Wrong pattern:**
```typescript
// NEVER do this - tests run against real project state
it('should work', async () => {
  const result = await runFeature();  // No isolation!
});
```

## Example Review Prompt

When spawning a subagent for review:

```
Review this implementation against the spec @spec-ref. Be strict:
- Every AC must have test coverage with // AC: annotation
- Missing tests are blocking issues, not suggestions
- Prioritize E2E tests over unit tests
- Verify tests run in isolated environments, not against real project state
- Reject fluff tests that don't validate real behavior
- List any issues as MUST-FIX
```

## Issue Severity

| Issue | Severity | Action |
|-------|----------|--------|
| Missing AC coverage | MUST-FIX | Add tests before PR |
| Fluff test | MUST-FIX | Rewrite or remove |
| Tests not isolated | MUST-FIX | Fix isolation |
| No E2E tests | Advisory | Consider adding |

## Integration

- **After task work**: Run local review before `/pr`
- **Before merge**: Complements `@pr-review-merge` workflow
- **In CI**: Automated review also runs but local catches issues earlier
