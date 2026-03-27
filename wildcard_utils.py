#!/usr/bin/env python3
"""
Wildcard Manager for HunyuanImage-3.0

Handles loading of wildcard definitions from JSON and random substitution
in prompts. Wildcards are specified using [key] syntax.

Features:
- Single wildcards: [animal] -> "tiger"
- Combined wildcards: [color+animal] -> "golden dragon"
- Multi-combine: [adjective+material+object] -> "ancient wooden chest"

Example:
    Prompt: "A [animal] in a [landscape] setting"
    Result: "A tiger in a forest setting"

    Prompt: "A [color+animal] in a [mood+landscape]"
    Result: "A silver wolf in a mysterious forest"
"""

import json
import random
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class WildcardManager:
    def __init__(self, json_path: str = "wildcards.json"):
        self.json_path = Path(json_path)
        self.data: Dict[str, List[str]] = {}
        self._load_data()

    def _load_data(self):
        """Load wildcard definitions from JSON file"""
        if not self.json_path.exists():
            print(f"Warning: {self.json_path} not found. Wildcards will not work.")
            return

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} wildcard categories")
        except Exception as e:
            print(f"Error loading wildcards: {e}")
            self.data = {}

    def reload(self):
        """Reload wildcards from file"""
        self._load_data()
        return f"Reloaded {len(self.data)} wildcard categories"

    def save(self):
        """Save current wildcard data to JSON file."""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get_available_wildcards(self) -> List[str]:
        """Returns a sorted list of all wildcard keys"""
        return sorted(list(self.data.keys()))

    def get_wildcard_count(self, key: str) -> int:
        """Get the number of items in a wildcard category"""
        return len(self.data.get(key, []))

    def get_wildcard_preview(self, key: str, count: int = 5) -> str:
        """Get a preview of items in a wildcard category"""
        items = self.data.get(key, [])
        if not items:
            return f"[{key}] - Not found"

        preview = items[:count]
        remaining = len(items) - count
        preview_str = ", ".join(preview)
        if remaining > 0:
            preview_str += f" ... and {remaining} more"
        return f"[{key}] ({len(items)} items): {preview_str}"

    def get_random_value(self, key: str) -> Optional[str]:
        """Get a random value from a wildcard category"""
        items = self.data.get(key, [])
        if items:
            return random.choice(items)
        return None

    def get_combined_value(self, combined_key: str, separator: str = " ") -> Optional[str]:
        """
        Get combined random values from multiple wildcard categories.

        Args:
            combined_key: Keys joined with '+', e.g., "color+animal"
            separator: String to join the values with (default: space)

        Returns:
            Combined string like "golden dragon" or None if any key is invalid

        Example:
            get_combined_value("color+animal") -> "silver wolf"
            get_combined_value("adjective+material+object") -> "ancient wooden chest"
        """
        keys = [k.strip() for k in combined_key.split('+')]

        # Check all keys exist
        values = []
        for key in keys:
            if key not in self.data:
                return None
            value = random.choice(self.data[key])
            values.append(value)

        return separator.join(values)

    def is_combined_wildcard(self, key: str) -> bool:
        """Check if a key uses the combination syntax (contains +)"""
        return '+' in key

    def is_alternating_wildcard(self, key: str) -> bool:
        """Check if a key uses the alternating syntax (contains /)"""
        return '/' in key

    def get_alternating_value(self, alternating_key: str, generation_index: int = 0) -> Optional[str]:
        """
        Get a value from an alternating wildcard based on generation index.

        Alternating wildcards use [option1/option2] syntax and switch between
        the options on each generation:
        - Even generations (0, 2, 4...): use first option
        - Odd generations (1, 3, 5...): use second option

        For more than 2 options, cycles through them:
        - [a/b/c]: index 0->a, 1->b, 2->c, 3->a, 4->b, ...

        Args:
            alternating_key: Keys joined with '/', e.g., "gems/materials"
            generation_index: Current generation number (0-based)

        Returns:
            Random value from the selected wildcard category, or None if invalid

        Example:
            get_alternating_value("gems/materials", 0) -> random gem
            get_alternating_value("gems/materials", 1) -> random material
            get_alternating_value("gems/materials", 2) -> random gem
        """
        options = [k.strip() for k in alternating_key.split('/')]

        if not options:
            return None

        # Cycle through options based on generation index
        selected_option = options[generation_index % len(options)]

        # The selected option could be a single wildcard or a combined wildcard
        if '+' in selected_option:
            return self.get_combined_value(selected_option)
        elif selected_option in self.data:
            return random.choice(self.data[selected_option])
        else:
            # Not a valid wildcard, return the literal text
            return selected_option

    def validate_combined_wildcard(self, combined_key: str) -> tuple:
        """
        Validate a combined wildcard and return info about it.

        Returns:
            (is_valid, missing_keys, found_keys)
        """
        keys = [k.strip() for k in combined_key.split('+')]
        missing = [k for k in keys if k not in self.data]
        found = [k for k in keys if k in self.data]
        return (len(missing) == 0, missing, found)

    def process_prompt(self, prompt: str, seed: Optional[int] = None,
                        generation_index: int = 0) -> str:
        """
        Process a prompt and replace all [key] wildcards with random values.

        Supports single, combined, and alternating wildcards:
        - [animal] -> picks one random animal
        - [color+animal] -> picks random color AND animal, joins with space
        - [adj+material+object] -> picks from all three, joins with spaces
        - [gems/materials] -> alternates between gems and materials each generation

        Args:
            prompt: The prompt text with [wildcard] placeholders
            seed: Optional random seed for reproducibility
            generation_index: Current generation number (for alternating wildcards)

        Returns:
            The processed prompt with wildcards replaced
        """
        if not prompt:
            return ""

        # Set seed if provided for reproducible results
        if seed is not None:
            random.seed(seed)

        # Regex to find [something] or [ something ]
        pattern = r"\[\s*([^\]]+?)\s*\]"

        def replace_match(match):
            key = match.group(1).strip()

            # Check for alternating wildcard syntax (e.g., "gems/materials")
            if '/' in key:
                alt_result = self.get_alternating_value(key, generation_index)
                if alt_result is not None:
                    return alt_result
                # If alternating fails, keep original
                return match.group(0)

            # Check for combined wildcard syntax (e.g., "color+animal")
            if '+' in key:
                combined_result = self.get_combined_value(key)
                if combined_result is not None:
                    return combined_result
                # If combined fails, keep original
                return match.group(0)

            # Check if single key exists in our data
            if key in self.data:
                choice = random.choice(self.data[key])
                return choice
            else:
                # Key not found, keep original to avoid breaking user prompt
                return match.group(0)

        # Keep replacing until no more known wildcards are found
        # (handles nested wildcards if a value contains another [wildcard])
        previous_prompt = ""
        max_iterations = 10  # Prevent infinite loops
        iterations = 0

        while prompt != previous_prompt and iterations < max_iterations:
            previous_prompt = prompt
            prompt = re.sub(pattern, replace_match, prompt)
            iterations += 1

        # Reset random seed to avoid affecting other random operations
        if seed is not None:
            random.seed()

        return prompt

    def process_prompt_batch(self, prompt: str, count: int,
                             base_seed: Optional[int] = None) -> List[str]:
        """
        Generate multiple variations of a prompt with different wildcard values.

        Args:
            prompt: The prompt text with [wildcard] placeholders
            count: Number of variations to generate
            base_seed: Optional base seed (each variation uses base_seed + index)

        Returns:
            List of processed prompts with different random values
        """
        results = []
        for i in range(count):
            seed = (base_seed + i) if base_seed is not None else None
            processed = self.process_prompt(prompt, seed=seed)
            results.append(processed)
        return results

    def _is_valid_wildcard_key(self, key: str) -> bool:
        """Check if a key (single, combined, or alternating) is valid"""
        key = key.strip()

        # Alternating wildcard - check all options are valid
        if '/' in key:
            options = [k.strip() for k in key.split('/')]
            for opt in options:
                if '+' in opt:
                    is_valid, _, _ = self.validate_combined_wildcard(opt)
                    if not is_valid:
                        return False
                elif opt not in self.data:
                    # Could be a literal string, which is allowed
                    pass
            return True

        # Combined wildcard - check all parts are valid
        if '+' in key:
            is_valid, _, _ = self.validate_combined_wildcard(key)
            return is_valid

        return key in self.data

    def has_alternating_wildcards(self, prompt: str) -> bool:
        """Check if a prompt contains any alternating wildcard syntax [a/b]"""
        if not prompt:
            return False
        pattern = r"\[\s*[^\]]*?/[^\]]*?\s*\]"
        return bool(re.search(pattern, prompt))

    def list_alternating_wildcards(self, prompt: str) -> List[str]:
        """List all alternating wildcards in a prompt"""
        if not prompt:
            return []
        pattern = r"\[\s*([^\]]+?)\s*\]"
        matches = re.findall(pattern, prompt)
        return [key.strip() for key in matches if '/' in key]

    def has_wildcards(self, prompt: str) -> bool:
        """Check if a prompt contains any wildcard syntax (single or combined)"""
        if not prompt:
            return False
        pattern = r"\[\s*([^\]]+?)\s*\]"
        matches = re.findall(pattern, prompt)
        # Check if any of the matches are valid wildcard keys (single or combined)
        return any(self._is_valid_wildcard_key(key) for key in matches)

    def list_wildcards_in_prompt(self, prompt: str) -> List[str]:
        """List all wildcards used in a prompt (single or combined)"""
        if not prompt:
            return []
        pattern = r"\[\s*([^\]]+?)\s*\]"
        matches = re.findall(pattern, prompt)
        return [key.strip() for key in matches if self._is_valid_wildcard_key(key)]

    def get_categories(self) -> Dict[str, List[str]]:
        """Group wildcards by category prefix"""
        categories = {}
        for key in self.data.keys():
            parts = key.split('-')
            if len(parts) > 1:
                category = parts[0]
            else:
                category = "general"

            if category not in categories:
                categories[category] = []
            categories[category].append(key)

        # Sort each category
        for cat in categories:
            categories[cat] = sorted(categories[cat])

        return categories

    def search_wildcards(self, query: str) -> List[str]:
        """Search for wildcards matching a query"""
        query = query.lower()
        return [key for key in self.data.keys() if query in key.lower()]

    # ============================================================
    # STARRED WILDCARD SYSTEM - Re-roll specific wildcards
    # ============================================================
    #
    # Starred wildcards use [*wildcard] syntax (asterisk prefix).
    # When processing:
    #   1. All wildcards are resolved initially
    #   2. Starred wildcards are tracked and can be re-rolled
    #   3. Same seed + same resolved prompt, but different starred values
    #
    # Example:
    #   Input:  "A [age] [*pose] girl in [setting]"
    #   First:  "A young dancing girl in forest" (seed 123)
    #   Reroll: "A young sitting girl in forest" (seed 123, only pose changed)
    # ============================================================

    def process_prompt_with_starred(self, prompt: str, seed: Optional[int] = None) -> Dict:
        """
        Process a prompt with starred wildcard support.

        Starred wildcards [*key] are resolved but tracked for re-rolling.

        Args:
            prompt: Prompt with [wildcard] and [*wildcard] placeholders
            seed: Optional seed for reproducibility

        Returns:
            dict with:
                - original_prompt: The input prompt
                - template_prompt: Prompt with only non-starred resolved, starred become [key]
                - full_prompt: Fully resolved prompt
                - starred_wildcards: List of starred wildcard keys
                - starred_values: Dict mapping key -> resolved value
                - non_starred_values: Dict mapping key -> resolved value (for reference)
        """
        if not prompt:
            return {
                "original_prompt": "",
                "template_prompt": "",
                "full_prompt": "",
                "starred_wildcards": [],
                "starred_values": {},
                "non_starred_values": {}
            }

        if seed is not None:
            random.seed(seed)

        starred_wildcards = []
        starred_values = {}
        non_starred_values = {}

        # Pattern to find [*key] or [key] or [ *key ] etc.
        pattern = r"\[\s*(\*?)([^\]]+?)\s*\]"

        def first_pass_replace(match):
            """First pass: resolve all wildcards, track starred ones"""
            is_starred = match.group(1) == '*'
            raw_key = match.group(2).strip()

            # Handle combined wildcards (e.g., "color+animal")
            if '+' in raw_key:
                value = self.get_combined_value(raw_key)
                if value is None:
                    return match.group(0)  # Keep original if invalid
            else:
                if raw_key not in self.data:
                    return match.group(0)  # Keep original if invalid
                value = random.choice(self.data[raw_key])

            if is_starred:
                starred_wildcards.append(raw_key)
                starred_values[raw_key] = value
            else:
                non_starred_values[raw_key] = value

            return value

        # Resolve all wildcards
        full_prompt = re.sub(pattern, first_pass_replace, prompt)

        # Handle nested wildcards (if a value contains another [wildcard])
        max_iterations = 10
        iterations = 0
        previous = ""
        while full_prompt != previous and iterations < max_iterations:
            previous = full_prompt
            full_prompt = re.sub(pattern, first_pass_replace, full_prompt)
            iterations += 1

        # Create template prompt: non-starred resolved, starred become [key] placeholder
        def template_replace(match):
            """Create template with starred as placeholders"""
            is_starred = match.group(1) == '*'
            raw_key = match.group(2).strip()

            if '+' in raw_key:
                value = self.get_combined_value(raw_key)
                if value is None:
                    return match.group(0)
            else:
                if raw_key not in self.data:
                    return match.group(0)
                # Use the same value we already picked
                if is_starred and raw_key in starred_values:
                    # Leave as placeholder for re-rolling
                    return f"[{raw_key}]"
                elif raw_key in non_starred_values:
                    value = non_starred_values[raw_key]
                else:
                    value = random.choice(self.data[raw_key])

            return value if not is_starred else f"[{raw_key}]"

        # Reset seed and create template
        if seed is not None:
            random.seed(seed)
        template_prompt = re.sub(pattern, template_replace, prompt)

        # Reset random seed
        if seed is not None:
            random.seed()

        return {
            "original_prompt": prompt,
            "template_prompt": template_prompt,
            "full_prompt": full_prompt,
            "starred_wildcards": starred_wildcards,
            "starred_values": starred_values,
            "non_starred_values": non_starred_values
        }

    def reroll_starred_wildcards(self, template_prompt: str,
                                  starred_wildcards: List[str]) -> Tuple[str, Dict[str, str]]:
        """
        Re-resolve only the starred wildcard placeholders in a template.

        Args:
            template_prompt: Prompt with [key] placeholders for starred wildcards
            starred_wildcards: List of wildcard keys that are starred

        Returns:
            Tuple of (resolved_prompt, new_values_dict)
        """
        new_values = {}

        pattern = r"\[\s*([^\]]+?)\s*\]"

        def replace_starred(match):
            key = match.group(1).strip()

            # Only replace if this is one of our starred wildcards
            if key in starred_wildcards:
                if '+' in key:
                    value = self.get_combined_value(key)
                    if value is None:
                        return match.group(0)
                else:
                    if key not in self.data:
                        return match.group(0)
                    value = random.choice(self.data[key])

                new_values[key] = value
                return value

            # Not a starred wildcard, keep as-is
            return match.group(0)

        resolved = re.sub(pattern, replace_starred, template_prompt)

        # Handle nested wildcards
        max_iterations = 10
        iterations = 0
        previous = ""
        while resolved != previous and iterations < max_iterations:
            previous = resolved
            resolved = re.sub(pattern, replace_starred, resolved)
            iterations += 1

        return resolved, new_values

    def has_starred_wildcards(self, prompt: str) -> bool:
        """Check if a prompt contains any starred wildcard syntax [*key]"""
        if not prompt:
            return False
        pattern = r"\[\s*\*[^\]]+?\s*\]"
        return bool(re.search(pattern, prompt))

    def list_starred_wildcards(self, prompt: str) -> List[str]:
        """List all starred wildcards in a prompt"""
        if not prompt:
            return []
        pattern = r"\[\s*\*([^\]]+?)\s*\]"
        matches = re.findall(pattern, prompt)
        return [key.strip() for key in matches]

    def generate_starred_variations(self, prompt: str, reroll_count: int = 5,
                                     seed: Optional[int] = None) -> List[Dict]:
        """
        Generate multiple variations by re-rolling only starred wildcards.

        This is the main entry point for the starred wildcard feature.

        Args:
            prompt: Prompt with [*wildcard] syntax for re-rollable wildcards
            reroll_count: How many variations to generate (including first)
            seed: Optional seed for the base prompt

        Returns:
            List of dicts, each containing:
                - prompt: The fully resolved prompt
                - seed: The seed used (same for all)
                - starred_values: Dict of starred wildcard -> value for this variation
                - variation_index: 0 for first, 1+ for re-rolls
        """
        # First, process the prompt with starred wildcard detection
        result = self.process_prompt_with_starred(prompt, seed=seed)

        if not result["starred_wildcards"]:
            # No starred wildcards, return single result
            return [{
                "prompt": result["full_prompt"],
                "seed": seed,
                "starred_values": {},
                "variation_index": 0
            }]

        variations = []

        # First variation is the original resolution
        variations.append({
            "prompt": result["full_prompt"],
            "seed": seed,
            "starred_values": result["starred_values"].copy(),
            "variation_index": 0
        })

        # Generate additional variations by re-rolling starred wildcards
        for i in range(1, reroll_count):
            new_prompt, new_values = self.reroll_starred_wildcards(
                result["template_prompt"],
                result["starred_wildcards"]
            )
            variations.append({
                "prompt": new_prompt,
                "seed": seed,  # Same seed for image generation
                "starred_values": new_values,
                "variation_index": i
            })

        return variations


# Global instance for easy import
wildcard_manager = WildcardManager(
    json_path=Path(__file__).parent / "wildcards.json"
)


def insert_wildcard(current_prompt: str, selected_wildcard: str) -> str:
    """Insert a wildcard tag into the prompt"""
    if not selected_wildcard:
        return current_prompt

    tag = f"[{selected_wildcard}]"
    if current_prompt:
        # Add space before if needed
        if not current_prompt.endswith(' '):
            return current_prompt + ' ' + tag
        return current_prompt + tag
    return tag


def preview_wildcard(key: str) -> str:
    """Get a preview of a wildcard category"""
    return wildcard_manager.get_wildcard_preview(key, count=8)


if __name__ == "__main__":
    # Test the wildcard manager
    print("Testing Wildcard Manager")
    print("=" * 50)

    print(f"\nAvailable wildcards: {len(wildcard_manager.get_available_wildcards())}")

    # Show some categories
    categories = wildcard_manager.get_categories()
    print(f"\nCategories: {list(categories.keys())[:10]}...")

    # Test prompt processing - single wildcards
    test_prompt = "A [animal] standing in a [landscape] with [weather] weather"
    print(f"\nTest prompt (single wildcards): {test_prompt}")

    for i in range(3):
        result = wildcard_manager.process_prompt(test_prompt)
        print(f"  Variation {i+1}: {result}")

    # Test combined wildcards
    print("\n" + "=" * 50)
    print("Testing Combined Wildcards (+ syntax)")
    print("=" * 50)

    # Find some valid wildcards to combine
    all_wildcards = wildcard_manager.get_available_wildcards()
    color_wc = [w for w in all_wildcards if 'color' in w.lower()][:1]
    animal_wc = [w for w in all_wildcards if 'animal' in w.lower()][:1]

    if color_wc and animal_wc:
        combined_prompt = f"A [{color_wc[0]}+{animal_wc[0]}] in a mystical forest"
        print(f"\nCombined prompt: {combined_prompt}")
        for i in range(3):
            result = wildcard_manager.process_prompt(combined_prompt)
            print(f"  Variation {i+1}: {result}")
    else:
        print("\nNote: Could not find color/animal wildcards for combined test")
        # Try a generic combined test with first two wildcards
        if len(all_wildcards) >= 2:
            combined_prompt = f"A [{all_wildcards[0]}+{all_wildcards[1]}] scene"
            print(f"\nCombined prompt: {combined_prompt}")
            for i in range(3):
                result = wildcard_manager.process_prompt(combined_prompt)
                print(f"  Variation {i+1}: {result}")

    # Test with seed for reproducibility
    print("\nWith seed=42:")
    for i in range(2):
        result = wildcard_manager.process_prompt(test_prompt, seed=42)
        print(f"  Run {i+1}: {result}")

    # Test starred wildcards
    print("\n" + "=" * 50)
    print("Testing Starred Wildcards [*key] (re-roll feature)")
    print("=" * 50)

    # Find a pose wildcard if available
    pose_wc = [w for w in all_wildcards if 'pose' in w.lower()]
    setting_wc = [w for w in all_wildcards if 'setting' in w.lower() or 'landscape' in w.lower()]

    if pose_wc:
        starred_prompt = f"A woman [*{pose_wc[0]}] in a beautiful [landscape]"
        print(f"\nStarred prompt: {starred_prompt}")
        print("(The [*pose] wildcard will be re-rolled while keeping the same setting)")

        # Check detection
        print(f"\nHas starred wildcards: {wildcard_manager.has_starred_wildcards(starred_prompt)}")
        print(f"Starred wildcards found: {wildcard_manager.list_starred_wildcards(starred_prompt)}")

        # Generate variations
        variations = wildcard_manager.generate_starred_variations(
            starred_prompt,
            reroll_count=4,
            seed=12345
        )

        print(f"\nGenerated {len(variations)} variations (same seed, different poses):")
        for var in variations:
            print(f"  [{var['variation_index']}] {var['prompt']}")
            print(f"      Starred values: {var['starred_values']}")
    else:
        # Fallback test with any available wildcard
        if len(all_wildcards) >= 2:
            starred_prompt = f"A [*{all_wildcards[0]}] with [{all_wildcards[1]}]"
            print(f"\nStarred prompt: {starred_prompt}")

            variations = wildcard_manager.generate_starred_variations(
                starred_prompt,
                reroll_count=3,
                seed=12345
            )

            print(f"\nGenerated {len(variations)} variations:")
            for var in variations:
                print(f"  [{var['variation_index']}] {var['prompt']}")

    # Test process_prompt_with_starred directly
    print("\n" + "-" * 30)
    print("Testing process_prompt_with_starred() details:")
    if pose_wc:
        test_starred = f"A young woman [*{pose_wc[0]}] in [landscape]"
    else:
        test_starred = f"A [*{all_wildcards[0]}] scene with [{all_wildcards[1]}]"

    result = wildcard_manager.process_prompt_with_starred(test_starred, seed=99)
    print(f"  Original: {result['original_prompt']}")
    print(f"  Template: {result['template_prompt']}")
    print(f"  Full:     {result['full_prompt']}")
    print(f"  Starred:  {result['starred_wildcards']}")
    print(f"  Values:   {result['starred_values']}")

    # Test alternating wildcards
    print("\n" + "=" * 50)
    print("Testing Alternating Wildcards [a/b] syntax")
    print("=" * 50)

    # Find two wildcards to alternate between
    if len(all_wildcards) >= 2:
        wc1, wc2 = all_wildcards[0], all_wildcards[1]
        alt_prompt = f"A scene with [{wc1}/{wc2}] in a beautiful setting"
        print(f"\nAlternating prompt: {alt_prompt}")
        print(f"  Will alternate between [{wc1}] and [{wc2}] each generation")
        print(f"\nHas alternating wildcards: {wildcard_manager.has_alternating_wildcards(alt_prompt)}")
        print(f"Alternating wildcards found: {wildcard_manager.list_alternating_wildcards(alt_prompt)}")

        print("\nGenerations 0-5 (alternating):")
        for gen_idx in range(6):
            result = wildcard_manager.process_prompt(alt_prompt, generation_index=gen_idx)
            which = wc1 if gen_idx % 2 == 0 else wc2
            print(f"  Gen {gen_idx} (uses {which}): {result}")

        # Test with 3 options
        if len(all_wildcards) >= 3:
            wc3 = all_wildcards[2]
            alt_prompt_3 = f"A [{wc1}/{wc2}/{wc3}] scene"
            print(f"\n3-way alternating: {alt_prompt_3}")
            for gen_idx in range(6):
                result = wildcard_manager.process_prompt(alt_prompt_3, generation_index=gen_idx)
                which = [wc1, wc2, wc3][gen_idx % 3]
                print(f"  Gen {gen_idx} (uses {which}): {result}")
