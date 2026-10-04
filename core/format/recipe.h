/* recipe.h -- which quantiser makes which weight, for rad-convert (spec §4.4).
 *
 * A recipe is ordered rules from a logical weight name to a quantiser and its options. The first
 * rule whose pattern matches a weight decides it; a weight no rule matches keeps the checkpoint's
 * encoding. Two spellings, one rule each:
 *
 *   in a file          PATTERN  QUANTISER  key=value key=value ...      # a comment
 *   on the command     PATTERN=QUANTISER:key=value,key=value
 *
 * A pattern is a glob over the whole name: `*` is any run of characters (dots included), `?` is
 * one. `$NAME` or `${NAME}` in an option is the environment's, expanded when the rule is read, so
 * one recipe serves every machine and the container records what it was actually given.
 */
#pragma once
#include "../rad_internal.h"

#include <string>
#include <utility>
#include <vector>

namespace rad {

struct RecipeRule {
    std::string pattern, quantizer;
    std::vector<std::pair<std::string, std::string>> options;   /* in the order written */
    std::string origin;                                         /* "file:12" or "--quant" */

    /* The options as one canonical string, `key=value` comma-separated and sorted by key: what a
     * container entry records and what --reuse compares, however the rule spelled them. */
    std::string options_text() const;
};

/* Does `name` match glob `pattern`? */
bool recipe_glob(const char* pattern, const char* name);

class Recipe {
public:
    /* Rules from a recipe file's text, appended. RAD_OK, or RAD_E_FORMAT with the line in `why`. */
    int add_text(const std::string& text, const std::string& origin, std::string* why);
    /* One rule in the command-line spelling, appended. */
    int add_flag(const std::string& spec, std::string* why);

    /* The rule that decides `name`, or null when none matches. */
    const RecipeRule* match(const std::string& name) const;

    const std::vector<RecipeRule>& rules() const { return rules_; }
    bool empty() const { return rules_.empty(); }
    /* Every rule in the file spelling, one a line, options expanded: what the container records. */
    std::string text() const;

private:
    std::vector<RecipeRule> rules_;
};

}  /* namespace rad */
