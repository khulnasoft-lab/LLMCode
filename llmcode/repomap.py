import colorsys
import math
import os
import random
import shutil
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from tqdm import tqdm

from llmcode.dump import dump
from llmcode.special import filter_important_files
from llmcode.waiting import Spinner

# Import new intelligent code analysis modules
from llmcode.static_analysis import StaticAnalyzer
from llmcode.code_structure import CodeStructureAnalyzer
from llmcode.relationship_mapper import RelationshipMapper
from llmcode.dependency_graph import DependencyGraphAnalyzer
from llmcode.file_selector import IntelligentFileSelector
from llmcode.relevance_scorer import RelevanceScorer
from llmcode.context_optimizer import ContextOptimizer
from llmcode.task_filter import TaskFilter
from llmcode.incremental_analyzer import IncrementalAnalyzer
from llmcode.parallel_processor import ParallelProcessor, ParallelAnalysisManager
from llmcode.memory_optimizer import MemoryOptimizationManager
from llmcode.persistence_layer import PersistenceManager

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from grep_ast.tsl import USING_TSL_PACK, get_language, get_parser  # noqa: E402

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError, OSError)


CACHE_VERSION = 3
if USING_TSL_PACK:
    CACHE_VERSION = 4

UPDATING_REPO_MAP_MESSAGE = "Updating repo map"


class RepoMap:
    TAGS_CACHE_DIR = f".llmcode.tags.cache.v{CACHE_VERSION}"

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=8,
        refresh="auto",
        enable_intelligent_analysis=True,
    ):
        self.io = io
        self.verbose = verbose
        self.refresh = refresh

        if not root:
            root = os.getcwd()
        self.root = root

        self.load_tags_cache()
        self.cache_threshold = 0.95

        self.max_map_tokens = map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window

        self.repo_content_prefix = repo_content_prefix

        self.main_model = main_model

        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_cache = {}
        self.map_processing_time = 0
        self.last_map = None
        
        # Initialize intelligent analysis modules
        self.enable_intelligent_analysis = enable_intelligent_analysis
        if self.enable_intelligent_analysis:
            self.static_analyzer = StaticAnalyzer()
            self.code_structure_analyzer = CodeStructureAnalyzer()
            self.relationship_mapper = RelationshipMapper()
            self.dependency_graph_analyzer = DependencyGraphAnalyzer()
            self.file_selector = IntelligentFileSelector()
            self.relevance_scorer = RelevanceScorer()
            self.context_optimizer = ContextOptimizer()
            self.task_filter = TaskFilter()
            self.incremental_analyzer = IncrementalAnalyzer()
            self.parallel_processor = ParallelProcessor()
            self.parallel_manager = ParallelAnalysisManager()
            self.memory_optimizer = MemoryOptimizationManager()
            self.persistence_manager = PersistenceManager()
            
            # Analysis result caches
            self.analysis_cache = {}
            self.relationship_cache = {}
            self.dependency_cache = {}
            self.structure_cache = {}
            
            if self.verbose:
                self.io.tool_output("Intelligent analysis modules initialized")

        if self.verbose:
            self.io.tool_output(
                f"RepoMap initialized with map_mul_no_files: {self.map_mul_no_files}"
            )

    def token_count(self, text):
        len_text = len(text)
        if len_text < 200:
            return self.main_model.token_count(text)

        lines = text.splitlines(keepends=True)
        num_lines = len(lines)
        step = num_lines // 100 or 1
        lines = lines[::step]
        sample_text = "".join(lines)
        sample_tokens = self.main_model.token_count(sample_text)
        est_tokens = sample_tokens / len(sample_text) * len_text
        return est_tokens

    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(
                int(max_map_tokens * self.map_mul_no_files),
                self.max_context_window - padding,
            )
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                force_refresh,
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        if self.verbose:
            num_tokens = self.token_count(files_listing)
            self.io.tool_output(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        try:
            return os.path.relpath(fname, self.root)
        except ValueError:
            # Issue #1288: ValueError: path is on mount 'C:', start on mount 'D:'
            # Just return the full fname.
            return fname

    def tags_cache_error(self, original_error=None):
        """Handle SQLite errors by trying to recreate cache, falling back to dict if needed"""

        if self.verbose and original_error:
            self.io.tool_warning(f"Tags cache error: {str(original_error)}")

        if isinstance(getattr(self, "TAGS_CACHE", None), dict):
            return

        path = Path(self.root) / self.TAGS_CACHE_DIR

        # Try to recreate the cache
        try:
            # Delete existing cache dir
            if path.exists():
                shutil.rmtree(path)

            # Try to create new cache
            new_cache = Cache(path)

            # Test that it works
            test_key = "test"
            new_cache[test_key] = "test"
            _ = new_cache[test_key]
            del new_cache[test_key]

            # If we got here, the new cache works
            self.TAGS_CACHE = new_cache
            return

        except SQLITE_ERRORS as e:
            # If anything goes wrong, warn and fall back to dict
            self.io.tool_warning(
                f"Unable to use tags cache at {path}, falling back to memory cache"
            )
            if self.verbose:
                self.io.tool_warning(f"Cache recreation error: {str(e)}")

        self.TAGS_CACHE = dict()

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        try:
            self.TAGS_CACHE = Cache(path)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_warning(f"File not found error: {fname}")

    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname
        try:
            val = self.TAGS_CACHE.get(cache_key)  # Issue #1308
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            val = self.TAGS_CACHE.get(cache_key)

        if val is not None and val.get("mtime") == file_mtime:
            try:
                return self.TAGS_CACHE[cache_key]["data"]
            except SQLITE_ERRORS as e:
                self.tags_cache_error(e)
                return self.TAGS_CACHE[cache_key]["data"]

        # miss!
        data = list(self.get_tags_raw(fname, rel_fname))

        # Update the cache
        try:
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
            self.save_tags_cache()
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}

        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            print(f"Skipping file {fname}: {err}")
            return

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        code = self.io.read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        saw = set()
        if USING_TSL_PACK:
            all_nodes = []
            for tag, nodes in captures.items():
                all_nodes += [(node, tag) for node in nodes]
        else:
            all_nodes = list(captures)

        for node, tag in all_nodes:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except Exception:  # On Windows, bad ref to time.clock which is deprecated?
            # self.io.tool_error(f"Error lexing {fname}")
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def get_ranked_tags(
        self,
        chat_fnames,
        other_fnames,
        mentioned_fnames,
        mentioned_idents,
        progress=None,
    ):
        import networkx as nx

        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 100 / len(fnames)

        try:
            cache_size = len(self.TAGS_CACHE)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            cache_size = len(self.TAGS_CACHE)

        if len(fnames) - cache_size > 100:
            self.io.tool_output(
                "Initial repo scan can be slow in larger repos, but only happens once."
            )
            fnames = tqdm(fnames, desc="Scanning repo")
            showing_bar = True
        else:
            showing_bar = False

        for fname in fnames:
            if self.verbose:
                self.io.tool_output(f"Processing {fname}")
            if progress and not showing_bar:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {fname}")

            try:
                file_ok = Path(fname).is_file()
            except OSError:
                file_ok = False

            if not file_ok:
                if fname not in self.warned_files:
                    self.io.tool_warning(f"Repo-map can't include {fname}")
                    self.io.tool_output(
                        "Has it been deleted from the file system but not from git?"
                    )
                    self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)
            current_pers = 0.0  # Start with 0 personalization score

            if fname in chat_fnames:
                current_pers += personalize
                chat_rel_fnames.add(rel_fname)

            if rel_fname in mentioned_fnames:
                # Use max to avoid double counting if in chat_fnames and mentioned_fnames
                current_pers = max(current_pers, personalize)

            # Check path components against mentioned_idents
            path_obj = Path(rel_fname)
            path_components = set(path_obj.parts)
            basename_with_ext = path_obj.name
            basename_without_ext, _ = os.path.splitext(basename_with_ext)
            components_to_check = path_components.union(
                {basename_with_ext, basename_without_ext}
            )

            matched_idents = components_to_check.intersection(mentioned_idents)
            if matched_idents:
                # Add personalization *once* if any path component matches a mentioned ident
                current_pers += personalize

            if current_pers > 0:
                personalization[rel_fname] = (
                    current_pers  # Assign the final calculated value
                )

            tags = list(self.get_tags(fname, rel_fname))
            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        ##
        # dump(defines)
        # dump(references)
        # dump(personalization)

        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        G = nx.MultiDiGraph()

        # Add a small self-edge for every definition that has no references
        # Helps with tree-sitter 0.23.2 with ruby, where "def greet(name)"
        # isn't counted as a def AND a ref. tree-sitter 0.24.0 does.
        for ident in defines.keys():
            if ident in references:
                continue
            for definer in defines[ident]:
                G.add_edge(definer, definer, weight=0.1, ident=ident)

        for ident in idents:
            if progress:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {ident}")

            definers = defines[ident]

            mul = 1.0

            is_snake = ("_" in ident) and any(c.isalpha() for c in ident)
            is_kebab = ("-" in ident) and any(c.isalpha() for c in ident)
            is_camel = any(c.isupper() for c in ident) and any(
                c.islower() for c in ident
            )
            if ident in mentioned_idents:
                mul *= 10
            if (is_snake or is_kebab or is_camel) and len(ident) >= 8:
                mul *= 10
            if ident.startswith("_"):
                mul *= 0.1
            if len(defines[ident]) > 5:
                mul *= 0.1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # dump(referencer, definer, num_refs, mul)
                    # if referencer == definer:
                    #    continue

                    use_mul = mul
                    if referencer in chat_rel_fnames:
                        use_mul *= 50

                    # scale down so high freq (low value) mentions don't dominate
                    num_refs = math.sqrt(num_refs)

                    G.add_edge(
                        referencer, definer, weight=use_mul * num_refs, ident=ident
                    )

        if not references:
            pass

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            # Issue #1536
            try:
                ranked = nx.pagerank(G, weight="weight")
            except ZeroDivisionError:
                return []

        # distribute the rank from each source node, across all of its out edges
        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            if progress:
                progress(f"{UPDATING_REPO_MAP_MESSAGE}: {src}")

            src_rank = ranked[src]
            total_weight = sum(
                data["weight"] for _src, _dst, data in G.out_edges(src, data=True)
            )
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: (x[1], x[0])
        )

        # dump(ranked_definitions)

        for (fname, ident), rank in ranked_definitions:
            # print(f"{rank:.03f} {fname} {ident}")
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(
            self.get_rel_fname(fname) for fname in other_fnames
        )

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted(
            [(rank, node) for (node, rank) in ranked.items()], reverse=True
        )
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        # Create a cache key
        cache_key = [
            tuple(sorted(chat_fnames)) if chat_fnames else None,
            tuple(sorted(other_fnames)) if other_fnames else None,
            max_map_tokens,
        ]

        if self.refresh == "auto":
            cache_key += [
                tuple(sorted(mentioned_fnames)) if mentioned_fnames else None,
                tuple(sorted(mentioned_idents)) if mentioned_idents else None,
            ]
        cache_key = tuple(cache_key)

        use_cache = False
        if not force_refresh:
            if self.refresh == "manual" and self.last_map:
                return self.last_map

            if self.refresh == "always":
                use_cache = False
            elif self.refresh == "files":
                use_cache = True
            elif self.refresh == "auto":
                use_cache = self.map_processing_time > 1.0

            # Check if the result is in the cache
            if use_cache and cache_key in self.map_cache:
                return self.map_cache[cache_key]

        # If not in cache or force_refresh is True, generate the map
        start_time = time.time()
        result = self.get_ranked_tags_map_uncached(
            chat_fnames,
            other_fnames,
            max_map_tokens,
            mentioned_fnames,
            mentioned_idents,
        )
        end_time = time.time()
        self.map_processing_time = end_time - start_time

        # Store the result in the cache
        self.map_cache[cache_key] = result
        self.last_map = result

        return result

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        spin = Spinner(UPDATING_REPO_MAP_MESSAGE)

        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            progress=spin.step,
        )

        other_rel_fnames = sorted(
            set(self.get_rel_fname(fname) for fname in other_fnames)
        )
        special_fnames = filter_important_files(other_rel_fnames)
        ranked_tags_fnames = set(tag[0] for tag in ranked_tags)
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_fnames = [(fn,) for fn in special_fnames]

        ranked_tags = special_fnames + ranked_tags

        spin.step()

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = set(self.get_rel_fname(fname) for fname in chat_fnames)

        self.tree_cache = dict()

        middle = min(int(max_map_tokens // 25), num_tags)
        while lower_bound <= upper_bound:
            # dump(lower_bound, middle, upper_bound)

            if middle > 1500:
                show_tokens = f"{middle / 1000.0:.1f}K"
            else:
                show_tokens = str(middle)
            spin.step(f"{UPDATING_REPO_MAP_MESSAGE}: {show_tokens} tokens")

            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.15
            if (
                num_tokens <= max_map_tokens and num_tokens > best_tree_tokens
            ) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens

                if pct_err < ok_err:
                    break

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = int((lower_bound + upper_bound) // 2)

        spin.end()
        return best_tree

    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        mtime = self.get_mtime(abs_fname)
        key = (rel_fname, tuple(sorted(lois)), mtime)

        if key in self.tree_cache:
            return self.tree_cache[key]

        if (
            rel_fname not in self.tree_context_cache
            or self.tree_context_cache[rel_fname]["mtime"] != mtime
        ):
            code = self.io.read_text(abs_fname) or ""
            if not code.endswith("\n"):
                code += "\n"

            context = TreeContext(
                rel_fname,
                code,
                color=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                # header_max=30,
                show_top_of_file_parent_scope=False,
            )
            self.tree_context_cache[rel_fname] = {"context": context, "mtime": mtime}

        context = self.tree_context_cache[rel_fname]["context"]
        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in sorted(tags) + [dummy_tag]:
            this_fname = tag.rel_fname
            if this_fname != cur_fname:
                if lois is not None:
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                lois = []
                cur_fname = this_fname
                cur_abs_fname = tag.fname

            lois.append(tag)

        return output

    # ========== INTELLIGENT ANALYSIS METHODS ==========
    
    def get_intelligent_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        task_type="code_review",
        query=None,
        force_refresh=False,
    ):
        """Get enhanced repository map with intelligent analysis capabilities."""
        if not self.enable_intelligent_analysis:
            return self.get_repo_map(
                chat_files, other_files, mentioned_fnames, mentioned_idents, force_refresh
            )
            
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
            
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()
            
        # Get intelligent file selection
        selected_files = self.file_selector.select_files(
            files=other_files,
            task_type=task_type,
            query=query,
            mentioned_files=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            max_tokens=self.max_map_tokens,
        )
        
        # Perform static analysis on selected files
        analysis_results = {}
        for file_path in selected_files:
            if file_path not in self.analysis_cache or force_refresh:
                try:
                    analysis = self.static_analyzer.analyze_file(file_path)
                    analysis_results[file_path] = analysis
                    self.analysis_cache[file_path] = analysis
                except Exception as e:
                    if self.verbose:
                        self.io.tool_warning(f"Failed to analyze {file_path}: {e}")
            else:
                analysis_results[file_path] = self.analysis_cache[file_path]
        
        # Get enhanced tags with intelligent analysis
        enhanced_tags = []
        for file_path in selected_files:
            rel_fname = self.get_rel_fname(file_path)
            tags = self.get_tags(file_path, rel_fname)
            
            # Enhance tags with analysis information
            for tag in tags:
                analysis = analysis_results.get(file_path, {})
                
                # Add analysis metadata to tag
                enhanced_tag = Tag(
                    rel_fname=tag.rel_fname,
                    fname=tag.fname,
                    line=tag.line,
                    name=f"{tag.name} [complexity:{analysis.get('complexity', 'unknown')}]",
                    kind=tag.kind,
                )
                enhanced_tags.append(enhanced_tag)
        
        # Apply task-aware filtering
        filtered_tags = self.task_filter.filter_tags(
            tags=enhanced_tags,
            task_type=task_type,
            query=query,
            max_tokens=self.max_map_tokens,
        )
        
        # Generate enhanced repo map
        files_listing = self.to_tree(filtered_tags, chat_files)
        
        if not files_listing:
            return
            
        if self.verbose:
            num_tokens = self.token_count(files_listing)
            self.io.tool_output(f"Intelligent repo-map: {num_tokens / 1024:.1f} k-tokens")
            
        if chat_files:
            other = "other "
        else:
            other = ""
            
        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""
            
        repo_content += files_listing
        
        return repo_content
    
    def get_cross_file_relationships(self, files: List[str], force_refresh: bool = False) -> Dict[str, Any]:
        """Get cross-file relationships and dependencies."""
        if not self.enable_intelligent_analysis:
            return {}
            
        cache_key = f"cross_file_relationships_{hash(tuple(sorted(files)))}"
        
        if cache_key in self.relationship_cache and not force_refresh:
            return self.relationship_cache[cache_key]
            
        try:
            # Analyze all files first
            analysis_results = {}
            for file_path in files:
                if file_path not in self.analysis_cache or force_refresh:
                    analysis = self.static_analyzer.analyze_file(file_path)
                    analysis_results[file_path] = analysis
                    self.analysis_cache[file_path] = analysis
                else:
                    analysis_results[file_path] = self.analysis_cache[file_path]
            
            # Get cross-file relationships
            relationships = self.relationship_mapper.map_cross_file_relationships(
                analysis_results
            )
            
            # Cache the result
            self.relationship_cache[cache_key] = relationships
            
            if self.verbose:
                self.io.tool_output(
                    f"Analyzed cross-file relationships for {len(files)} files"
                )
                
            return relationships
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to get cross-file relationships: {e}")
            return {}
    
    def get_dependency_analysis(self, files: List[str], force_refresh: bool = False) -> Dict[str, Any]:
        """Get comprehensive dependency analysis for files."""
        if not self.enable_intelligent_analysis:
            return {}
            
        cache_key = f"dependency_analysis_{hash(tuple(sorted(files)))}"
        
        if cache_key in self.dependency_cache and not force_refresh:
            return self.dependency_cache[cache_key]
            
        try:
            # Build dependency graph
            dependency_graph = self.dependency_graph_analyzer.build_dependency_graph(files)
            
            # Analyze dependencies
            analysis = {
                "dependency_graph": dependency_graph,
                "circular_dependencies": self.dependency_graph_analyzer.detect_circular_dependencies(dependency_graph),
                "module_clusters": self.dependency_graph_analyzer.cluster_modules(dependency_graph),
                "dependency_stats": self.dependency_graph_analyzer.get_dependency_statistics(dependency_graph),
            }
            
            # Cache the result
            self.dependency_cache[cache_key] = analysis
            
            if self.verbose:
                self.io.tool_output(
                    f"Analyzed dependencies for {len(files)} files, "
                    f"found {len(analysis['circular_dependencies'])} circular dependencies"
                )
                
            return analysis
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to get dependency analysis: {e}")
            return {}
    
    def get_semantic_search_results(
        self,
        query: str,
        files: List[str],
        max_results: int = 10,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search across files."""
        if not self.enable_intelligent_analysis:
            return []
            
        try:
            # Get relevance scores for all files
            relevance_results = []
            for file_path in files:
                try:
                    # Get file analysis if not cached
                    if file_path not in self.analysis_cache or force_refresh:
                        analysis = self.static_analyzer.analyze_file(file_path)
                        self.analysis_cache[file_path] = analysis
                    else:
                        analysis = self.analysis_cache[file_path]
                    
                    # Calculate relevance score
                    relevance_score = self.relevance_scorer.calculate_relevance(
                        code_section=analysis,
                        query=query,
                        task_type="semantic_search",
                        context={"file_path": file_path},
                    )
                    
                    relevance_results.append({
                        "file_path": file_path,
                        "relevance_score": relevance_score.score,
                        "confidence": relevance_score.confidence,
                        "explanation": relevance_score.explanation,
                        "analysis": analysis,
                    })
                    
                except Exception as e:
                    if self.verbose:
                        self.io.tool_warning(f"Failed to analyze {file_path} for semantic search: {e}")
            
            # Sort by relevance score and return top results
            relevance_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return relevance_results[:max_results]
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to perform semantic search: {e}")
            return []
    
    def optimize_context_window(
        self,
        content: str,
        max_tokens: int,
        strategy: str = "hybrid",
        task_type: str = "general",
    ) -> str:
        """Optimize content for context window using various strategies."""
        if not self.enable_intelligent_analysis:
            return content
            
        try:
            # Calculate current token count
            current_tokens = self.token_count(content)
            
            if current_tokens <= max_tokens:
                return content
            
            # Apply context optimization
            optimized_content = self.context_optimizer.optimize_context(
                content=content,
                max_tokens=max_tokens,
                strategy=strategy,
                task_type=task_type,
            )
            
            if self.verbose:
                original_tokens = current_tokens
                optimized_tokens = self.token_count(optimized_content)
                reduction = ((original_tokens - optimized_tokens) / original_tokens) * 100
                self.io.tool_output(
                    f"Context optimization: {original_tokens} -> {optimized_tokens} tokens "
                    f"({reduction:.1f}% reduction)"
                )
            
            return optimized_content
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to optimize context: {e}")
            return content
    
    def get_analysis_summary(self, files: List[str]) -> Dict[str, Any]:
        """Get a comprehensive summary of analysis results for files."""
        if not self.enable_intelligent_analysis:
            return {}
            
        summary = {
            "total_files": len(files),
            "analyzed_files": 0,
            "complexity_stats": {},
            "relationship_stats": {},
            "dependency_stats": {},
            "file_types": {},
        }
        
        try:
            for file_path in files:
                if file_path in self.analysis_cache:
                    summary["analyzed_files"] += 1
                    analysis = self.analysis_cache[file_path]
                    
                    # Collect complexity stats
                    complexity = analysis.get("complexity", "unknown")
                    summary["complexity_stats"][complexity] = summary["complexity_stats"].get(complexity, 0) + 1
                    
                    # Collect file types
                    file_type = analysis.get("language", "unknown")
                    summary["file_types"][file_type] = summary["file_types"].get(file_type, 0) + 1
            
            # Get relationship stats if available
            cache_key = f"cross_file_relationships_{hash(tuple(sorted(files)))}"
            if cache_key in self.relationship_cache:
                relationships = self.relationship_cache[cache_key]
                summary["relationship_stats"] = {
                    "total_relationships": len(relationships.get("relationships", [])),
                    "relationship_types": {},
                }
                
                for rel in relationships.get("relationships", []):
                    rel_type = rel.get("type", "unknown")
                    summary["relationship_stats"]["relationship_types"][rel_type] = \
                        summary["relationship_stats"]["relationship_types"].get(rel_type, 0) + 1
            
            # Get dependency stats if available
            cache_key = f"dependency_analysis_{hash(tuple(sorted(files)))}"
            if cache_key in self.dependency_cache:
                dependency_analysis = self.dependency_cache[cache_key]
                summary["dependency_stats"] = dependency_analysis.get("dependency_stats", {})
            
            return summary
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to generate analysis summary: {e}")
            return summary
    
    def get_incremental_analysis(
        self,
        file_paths,
        analysis_types=None,
        force_refresh=False,
    ):
        """Get incremental analysis results for files.
        
        Args:
            file_paths: List of file paths to analyze
            analysis_types: Set of analysis types to perform (default: all)
            force_refresh: Whether to force refresh all analysis
            
        Returns:
            Dictionary containing incremental analysis results
        """
        if not self.enable_intelligent_analysis:
            return {
                'results': {},
                'delta': {
                    'added_files': [],
                    'modified_files': [],
                    'deleted_files': [],
                    'affected_dependencies': []
                },
                'files_analyzed': 0,
                'total_files': len(file_paths),
                'timestamp': time.time(),
                'error': 'Intelligent analysis not enabled'
            }
        
        if analysis_types is None:
            analysis_types = {'static', 'structure', 'dependency', 'relationship'}
        
        try:
            # Perform incremental analysis
            results = self.incremental_analyzer.analyze_incremental(
                file_paths=file_paths,
                analysis_types=analysis_types,
                force_refresh=force_refresh,
            )
            
            if self.verbose:
                delta = results['delta']
                self.io.tool_output(
                    f"Incremental analysis completed: {results['files_analyzed']}/{results['total_files']} files analyzed"
                )
                if delta.get('added_files'):
                    self.io.tool_output(f"  Added: {len(delta['added_files'])} files")
                if delta.get('modified_files'):
                    self.io.tool_output(f"  Modified: {len(delta['modified_files'])} files")
                if delta.get('deleted_files'):
                    self.io.tool_output(f"  Deleted: {len(delta['deleted_files'])} files")
                if delta.get('affected_dependencies'):
                    self.io.tool_output(f"  Affected dependencies: {len(delta['affected_dependencies'])} files")
            
            return results
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to perform incremental analysis: {e}")
            return {
                'results': {},
                'delta': {
                    'added_files': [],
                    'modified_files': [],
                    'deleted_files': [],
                    'affected_dependencies': []
                },
                'files_analyzed': 0,
                'total_files': len(file_paths),
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def get_incremental_analysis_stats(self):
        """Get statistics about incremental analysis.
        
        Returns:
            Dictionary containing analysis statistics
        """
        if not self.enable_intelligent_analysis:
            return {'error': 'Intelligent analysis not enabled'}
        
        try:
            return self.incremental_analyzer.get_analysis_stats()
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to get incremental analysis stats: {e}")
            return {'error': str(e)}
    
    def clear_incremental_cache(self):
        """Clear incremental analysis cache."""
        if not self.enable_intelligent_analysis:
            return
        
        try:
            self.incremental_analyzer.clear_cache()
            if self.verbose:
                self.io.tool_output("Incremental analysis cache cleared")
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to clear incremental cache: {e}")
    
    def invalidate_incremental_file(self, file_path):
        """Invalidate incremental analysis cache for a specific file.
        
        Args:
            file_path: Path to file to invalidate
        """
        if not self.enable_intelligent_analysis:
            return
        
        try:
            self.incremental_analyzer.invalidate_file(file_path)
            if self.verbose:
                self.io.tool_output(f"Invalidated incremental cache for {file_path}")
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to invalidate file {file_path}: {e}")
    
    def get_parallel_analysis(
        self,
        file_paths,
        analysis_types=None,
        max_workers=None,
        use_processes=False,
        force_refresh=False,
    ):
        """Get parallel analysis results for files.
        
        Args:
            file_paths: List of file paths to analyze
            analysis_types: Set of analysis types to perform (default: all)
            max_workers: Maximum number of parallel workers
            use_processes: Whether to use processes instead of threads
            force_refresh: Whether to force refresh all analysis
            
        Returns:
            Dictionary containing parallel analysis results and statistics
        """
        if not self.enable_intelligent_analysis:
            return {
                'results': {},
                'stats': {
                    'total_tasks': 0,
                    'completed_tasks': 0,
                    'failed_tasks': 0,
                    'total_time': 0,
                    'average_time_per_task': 0,
                    'peak_memory_usage': 0,
                    'cpu_utilization': 0,
                    'throughput': 0
                },
                'error': 'Intelligent analysis not enabled'
            }
        
        if analysis_types is None:
            analysis_types = {'static', 'structure', 'dependency', 'relationship'}
        
        try:
            # Configure parallel processor
            if max_workers is not None:
                self.parallel_processor.max_workers = max_workers
            self.parallel_processor.use_processes = use_processes
            
            # Process files in parallel
            results_by_file, stats = self.parallel_processor.process_files(
                file_paths=file_paths,
                analysis_types=analysis_types
            )
            
            # Update caches with successful results
            for file_path, result in results_by_file.items():
                if result.success and not force_refresh:
                    if 'static' in result.results:
                        self.analysis_cache[file_path] = result.results['static']
                    if 'structure' in result.results:
                        self.structure_cache[file_path] = result.results['structure']
                    if 'dependencies' in result.results:
                        self.dependency_cache[file_path] = result.results['dependencies']
                    if 'relationships' in result.results:
                        self.relationship_cache[file_path] = result.results['relationships']
            
            if self.verbose:
                self.io.tool_output(
                    f"Parallel analysis completed: {stats.completed_tasks}/{stats.total_tasks} tasks completed"
                )
                self.io.tool_output(f"  Total time: {stats.total_time:.2f}s")
                self.io.tool_output(f"  Average time per task: {stats.average_time_per_task:.3f}s")
                self.io.tool_output(f"  Throughput: {stats.throughput:.2f} tasks/s")
                if stats.failed_tasks > 0:
                    self.io.tool_output(f"  Failed tasks: {stats.failed_tasks}")
            
            return {
                'results': {file_path: result.results for file_path, result in results_by_file.items()},
                'stats': {
                    'total_tasks': stats.total_tasks,
                    'completed_tasks': stats.completed_tasks,
                    'failed_tasks': stats.failed_tasks,
                    'total_time': stats.total_time,
                    'average_time_per_task': stats.average_time_per_task,
                    'peak_memory_usage': stats.peak_memory_usage,
                    'cpu_utilization': stats.cpu_utilization,
                    'throughput': stats.throughput
                },
                'success_rate': stats.completed_tasks / max(1, stats.total_tasks)
            }
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to perform parallel analysis: {e}")
            return {
                'results': {},
                'stats': {
                    'total_tasks': len(file_paths),
                    'completed_tasks': 0,
                    'failed_tasks': len(file_paths),
                    'total_time': 0,
                    'average_time_per_task': 0,
                    'peak_memory_usage': 0,
                    'cpu_utilization': 0,
                    'throughput': 0
                },
                'error': str(e),
                'success_rate': 0
            }
    
    def analyze_repository_parallel(
        self,
        root_dir=None,
        file_patterns=None,
        analysis_types=None,
        exclude_patterns=None,
        max_workers=None,
        use_processes=False,
    ):
        """Analyze an entire repository in parallel.
        
        Args:
            root_dir: Root directory to analyze (default: self.root)
            file_patterns: List of file patterns to include
            analysis_types: Set of analysis types to perform
            exclude_patterns: List of patterns to exclude
            max_workers: Maximum number of parallel workers
            use_processes: Whether to use processes instead of threads
            
        Returns:
            Dictionary containing repository analysis results and statistics
        """
        if not self.enable_intelligent_analysis:
            return {
                'results': {},
                'stats': {
                    'total_tasks': 0,
                    'completed_tasks': 0,
                    'failed_tasks': 0,
                    'total_time': 0,
                    'average_time_per_task': 0,
                    'peak_memory_usage': 0,
                    'cpu_utilization': 0,
                    'throughput': 0
                },
                'error': 'Intelligent analysis not enabled'
            }
        
        if root_dir is None:
            root_dir = self.root
        
        if analysis_types is None:
            analysis_types = {'static', 'structure', 'dependency', 'relationship'}
        
        try:
            # Configure parallel manager
            if max_workers is not None:
                self.parallel_manager.processor.max_workers = max_workers
            self.parallel_manager.processor.use_processes = use_processes
            
            # Analyze repository
            results, stats = self.parallel_manager.analyze_repository(
                root_dir=root_dir,
                file_patterns=file_patterns,
                analysis_types=analysis_types,
                exclude_patterns=exclude_patterns
            )
            
            if self.verbose:
                self.io.tool_output(
                    f"Repository analysis completed: {stats.completed_tasks}/{stats.total_tasks} files analyzed"
                )
                self.io.tool_output(f"  Total time: {stats.total_time:.2f}s")
                self.io.tool_output(f"  Average time per file: {stats.average_time_per_task:.3f}s")
                self.io.tool_output(f"  Throughput: {stats.throughput:.2f} files/s")
                if stats.failed_tasks > 0:
                    self.io.tool_output(f"  Failed files: {stats.failed_tasks}")
            
            return {
                'results': results,
                'stats': {
                    'total_tasks': stats.total_tasks,
                    'completed_tasks': stats.completed_tasks,
                    'failed_tasks': stats.failed_tasks,
                    'total_time': stats.total_time,
                    'average_time_per_task': stats.average_time_per_task,
                    'peak_memory_usage': stats.peak_memory_usage,
                    'cpu_utilization': stats.cpu_utilization,
                    'throughput': stats.throughput
                },
                'success_rate': stats.completed_tasks / max(1, stats.total_tasks)
            }
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to analyze repository in parallel: {e}")
            return {
                'results': {},
                'stats': {
                    'total_tasks': 0,
                    'completed_tasks': 0,
                    'failed_tasks': 0,
                    'total_time': 0,
                    'average_time_per_task': 0,
                    'peak_memory_usage': 0,
                    'cpu_utilization': 0,
                    'throughput': 0
                },
                'error': str(e),
                'success_rate': 0
            }
    
    def get_parallel_processing_info(self):
        """Get information about parallel processing capabilities.
        
        Returns:
            Dictionary containing parallel processing information
        """
        if not self.enable_intelligent_analysis:
            return {'error': 'Intelligent analysis not enabled'}
        
        try:
            system_info = self.parallel_processor.get_system_info()
            manager_info = {
                'auto_tune': self.parallel_manager.auto_tune,
                'processor_configured': True
            }
            
            return {
                'system_info': system_info,
                'manager_info': manager_info,
                'available': True
            }
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to get parallel processing info: {e}")
            return {'error': str(e), 'available': False}
    
    def estimate_parallel_processing_time(self, num_files, avg_file_size_kb=10):
        """Estimate parallel processing time for a given number of files.
        
        Args:
            num_files: Number of files to process
            avg_file_size_kb: Average file size in KB
            
        Returns:
            Estimated processing time in seconds
        """
        if not self.enable_intelligent_analysis:
            return 0
        
        try:
            return self.parallel_processor.estimate_processing_time(num_files, avg_file_size_kb)
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to estimate processing time: {e}")
            return 0
    
    def analyze_with_memory_optimization(self, file_paths):
        """Analyze files with memory optimization.
        
        Args:
            file_paths: List of file paths to analyze
            
        Returns:
            Dictionary containing analysis results with memory optimization
        """
        if not self.enable_intelligent_analysis:
            return {
                'files': [],
                'dependency_graph': {},
                'memory_stats': {},
                'optimization_stats': {},
                'total_files': 0,
                'successful_files': 0,
                'failed_files': 0,
                'error': 'Intelligent analysis not enabled'
            }
        
        try:
            # Analyze files with memory optimization
            results = self.memory_optimizer.analyze_with_memory_optimization(file_paths)
            
            # Update caches with successful results
            for file_result in results.get('files', []):
                if 'error' not in file_result:
                    file_path = file_result['file_path']
                    if 'static' in file_result:
                        self.analysis_cache[file_path] = file_result['static']
                    if 'structure' in file_result:
                        self.structure_cache[file_path] = file_result['structure']
                    if 'dependencies' in file_result:
                        self.dependency_cache[file_path] = file_result['dependencies']
                    if 'relationships' in file_result:
                        self.relationship_cache[file_path] = file_result['relationships']
            
            if self.verbose:
                self.io.tool_output(
                    f"Memory-optimized analysis completed: {results.get('successful_files', 0)}/{results.get('total_files', 0)} files analyzed"
                )
                memory_stats = results.get('memory_stats', {})
                if 'peak_rss_mb' in memory_stats:
                    self.io.tool_output(f"  Peak memory usage: {memory_stats['peak_rss_mb']:.2f} MB")
                
                optimization_stats = results.get('optimization_stats', {})
                if 'optimization_stats' in optimization_stats:
                    opt_stats = optimization_stats['optimization_stats']
                    if 'compressions' in opt_stats and opt_stats['compressions'] > 0:
                        self.io.tool_output(f"  Graph compressions: {opt_stats['compressions']}")
                    if 'sparsifications' in opt_stats and opt_stats['sparsifications'] > 0:
                        self.io.tool_output(f"  Graph sparsifications: {opt_stats['sparsifications']}")
                    if 'memory_savings_mb' in opt_stats and opt_stats['memory_savings_mb'] > 0:
                        self.io.tool_output(f"  Memory savings: {opt_stats['memory_savings_mb']:.2f} MB")
            
            return results
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to analyze with memory optimization: {e}")
            return {
                'files': [],
                'dependency_graph': {},
                'memory_stats': {},
                'optimization_stats': {},
                'total_files': len(file_paths),
                'successful_files': 0,
                'failed_files': len(file_paths),
                'error': str(e)
            }
    
    def optimize_dependency_graph(self, graph):
        """Optimize a dependency graph for memory usage.
        
        Args:
            graph: Dependency graph to optimize
            
        Returns:
            Optimized dependency graph
        """
        if not self.enable_intelligent_analysis:
            return graph
        
        try:
            optimized_graph = self.memory_optimizer.optimize_dependency_graph(graph)
            
            if self.verbose:
                original_size = len(graph)
                optimized_size = len(optimized_graph)
                if original_size != optimized_size:
                    self.io.tool_output(
                        f"Dependency graph optimized: {original_size} -> {optimized_size} nodes"
                    )
            
            return optimized_graph
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to optimize dependency graph: {e}")
            return graph
    
    def get_memory_report(self):
        """Get comprehensive memory report.
        
        Returns:
            Dictionary containing memory report
        """
        if not self.enable_intelligent_analysis:
            return {'error': 'Intelligent analysis not enabled'}
        
        try:
            report = self.memory_optimizer.get_memory_report()
            
            if self.verbose:
                memory_stats = report.get('memory_stats', {})
                memory_usage = memory_stats.get('memory_usage', {})
                if 'current_rss_mb' in memory_usage:
                    self.io.tool_output(f"Current memory usage: {memory_usage['current_rss_mb']:.2f} MB")
                if 'peak_rss_mb' in memory_usage:
                    self.io.tool_output(f"Peak memory usage: {memory_usage['peak_rss_mb']:.2f} MB")
                
                optimization_stats = report.get('optimization_stats', {})
                if 'optimization_stats' in optimization_stats:
                    opt_stats = optimization_stats['optimization_stats']
                    if 'memory_savings_mb' in opt_stats:
                        self.io.tool_output(f"Total memory savings: {opt_stats['memory_savings_mb']:.2f} MB")
            
            return report
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to get memory report: {e}")
            return {'error': str(e)}
    
    def clear_memory_caches(self):
        """Clear all memory caches to free memory.
        
        Returns:
            Dictionary containing operation status
        """
        if not self.enable_intelligent_analysis:
            return {'error': 'Intelligent analysis not enabled'}
        
        try:
            self.memory_optimizer.clear_all_caches()
            
            if self.verbose:
                self.io.tool_output("All memory caches cleared")
            
            return {'success': True, 'message': 'Memory caches cleared'}
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to clear memory caches: {e}")
            return {'error': str(e), 'success': False}
    
    def set_memory_limit(self, memory_limit_mb):
        """Set memory limit for optimization.
        
        Args:
            memory_limit_mb: Memory limit in MB
            
        Returns:
            Dictionary containing operation status
        """
        if not self.enable_intelligent_analysis:
            return {'error': 'Intelligent analysis not enabled'}
        
        try:
            self.memory_optimizer.max_memory_mb = memory_limit_mb
            self.memory_optimizer.memory_analyzer.max_memory_mb = memory_limit_mb
            
            if self.verbose:
                self.io.tool_output(f"Memory limit set to {memory_limit_mb} MB")
            
            return {'success': True, 'memory_limit_mb': memory_limit_mb}
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to set memory limit: {e}")
            return {'error': str(e), 'success': False}
    
    def get_memory_usage_stats(self):
        """Get current memory usage statistics.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        if not self.enable_intelligent_analysis:
            return {'error': 'Intelligent analysis not enabled'}
        
        try:
            current_memory = self.memory_optimizer.memory_analyzer.memory_monitor.get_current_memory()
            peak_memory = self.memory_optimizer.memory_analyzer.memory_monitor.get_peak_memory()
            
            stats = {
                'current_memory_mb': current_memory.current_rss / (1024 * 1024),
                'peak_memory_mb': peak_memory.current_rss / (1024 * 1024),
                'current_vms_mb': current_memory.current_vms / (1024 * 1024),
                'peak_vms_mb': peak_memory.current_vms / (1024 * 1024),
                'memory_limit_mb': self.memory_optimizer.max_memory_mb,
                'memory_pressure': current_memory.current_rss > (self.memory_optimizer.max_memory_mb * 1024 * 1024 * 0.8)
            }
            
            if self.verbose:
                self.io.tool_output(f"Current memory usage: {stats['current_memory_mb']:.2f} MB")
                self.io.tool_output(f"Memory limit: {stats['memory_limit_mb']} MB")
                if stats['memory_pressure']:
                    self.io.tool_output("Memory pressure detected")
            
            return stats
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to get memory usage stats: {e}")
            return {'error': str(e)}
    
    def store_analysis_result(self, file_path, analysis_type, result, ttl_hours=None):
        """Store analysis result in persistence layer.
        
        Args:
            file_path: Path to the analyzed file
            analysis_type: Type of analysis performed
            result: Analysis result data
            ttl_hours: Time-to-live in hours (optional)
            
        Returns:
            Success status
        """
        if not self.enable_intelligent_analysis:
            return False
        
        try:
            success = self.persistence_manager.store_analysis_result(
                file_path=file_path,
                analysis_type=analysis_type,
                result=result,
                ttl_hours=ttl_hours
            )
            
            if self.verbose and success:
                self.io.tool_output(f"Stored {analysis_type} analysis result for {file_path}")
            
            return success
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to store analysis result: {e}")
            return False
    
    def retrieve_analysis_result(self, file_path, analysis_type):
        """Retrieve analysis result from persistence layer.
        
        Args:
            file_path: Path to the analyzed file
            analysis_type: Type of analysis performed
            
        Returns:
            Analysis result or None if not found
        """
        if not self.enable_intelligent_analysis:
            return None
        
        try:
            result = self.persistence_manager.retrieve_analysis_result(
                file_path=file_path,
                analysis_type=analysis_type
            )
            
            if self.verbose and result:
                self.io.tool_output(f"Retrieved {analysis_type} analysis result for {file_path}")
            
            return result
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to retrieve analysis result: {e}")
            return None
    
    def delete_analysis_result(self, file_path, analysis_type):
        """Delete analysis result from persistence layer.
        
        Args:
            file_path: Path to the analyzed file
            analysis_type: Type of analysis performed
            
        Returns:
            Success status
        """
        if not self.enable_intelligent_analysis:
            return False
        
        try:
            success = self.persistence_manager.delete_analysis_result(
                file_path=file_path,
                analysis_type=analysis_type
            )
            
            if self.verbose and success:
                self.io.tool_output(f"Deleted {analysis_type} analysis result for {file_path}")
            
            return success
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to delete analysis result: {e}")
            return False
    
    def list_analysis_results(self, analysis_type=None):
        """List analysis results from persistence layer.
        
        Args:
            analysis_type: Filter by analysis type (optional)
            
        Returns:
            List of analysis result metadata
        """
        if not self.enable_intelligent_analysis:
            return []
        
        try:
            results = self.persistence_manager.list_analysis_results(analysis_type=analysis_type)
            
            if self.verbose:
                self.io.tool_output(f"Found {len(results)} analysis results")
                if analysis_type:
                    self.io.tool_output(f"  Filtered by type: {analysis_type}")
            
            return results
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to list analysis results: {e}")
            return []
    
    def clear_all_analysis_results(self):
        """Clear all analysis results from persistence layer.
        
        Returns:
            Success status
        """
        if not self.enable_intelligent_analysis:
            return False
        
        try:
            success = self.persistence_manager.clear_all_results()
            
            if self.verbose and success:
                self.io.tool_output("Cleared all analysis results")
            
            return success
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to clear analysis results: {e}")
            return False
    
    def get_persistence_stats(self):
        """Get comprehensive persistence statistics.
        
        Returns:
            Dictionary containing persistence statistics
        """
        if not self.enable_intelligent_analysis:
            return {'error': 'Intelligent analysis not enabled'}
        
        try:
            stats = self.persistence_manager.get_persistence_stats()
            
            if self.verbose:
                cache_stats = stats.get('cache_stats', {})
                if 'hit_rate' in cache_stats:
                    self.io.tool_output(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
                if 'memory_cache_size' in cache_stats:
                    self.io.tool_output(f"Memory cache size: {cache_stats['memory_cache_size']}")
                
                storage_info = cache_stats.get('storage_info', {})
                if 'total_size_mb' in storage_info:
                    self.io.tool_output(f"Storage size: {storage_info['total_size_mb']:.2f} MB")
            
            return stats
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to get persistence stats: {e}")
            return {'error': str(e)}
    
    def cleanup_expired_results(self):
        """Clean up expired analysis results.
        
        Returns:
            Number of cleaned results
        """
        if not self.enable_intelligent_analysis:
            return 0
        
        try:
            cleaned_count = self.persistence_manager.cleanup_expired_results()
            
            if self.verbose and cleaned_count > 0:
                self.io.tool_output(f"Cleaned up {cleaned_count} expired analysis results")
            
            return cleaned_count
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to cleanup expired results: {e}")
            return 0
    
    def export_analysis_results(self, export_path, analysis_type=None):
        """Export analysis results to file.
        
        Args:
            export_path: Path to export file
            analysis_type: Filter by analysis type (optional)
            
        Returns:
            Success status
        """
        if not self.enable_intelligent_analysis:
            return False
        
        try:
            success = self.persistence_manager.export_results(
                export_path=export_path,
                analysis_type=analysis_type
            )
            
            if self.verbose and success:
                self.io.tool_output(f"Exported analysis results to {export_path}")
                if analysis_type:
                    self.io.tool_output(f"  Filtered by type: {analysis_type}")
            
            return success
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to export analysis results: {e}")
            return False
    
    def import_analysis_results(self, import_path):
        """Import analysis results from file.
        
        Args:
            import_path: Path to import file
            
        Returns:
            Success status
        """
        if not self.enable_intelligent_analysis:
            return False
        
        try:
            success = self.persistence_manager.import_results(import_path=import_path)
            
            if self.verbose and success:
                self.io.tool_output(f"Imported analysis results from {import_path}")
            
            return success
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to import analysis results: {e}")
            return False
    
    def get_cached_analysis(self, file_path, analysis_type):
        """Get cached analysis result, checking persistence layer first.
        
        Args:
            file_path: Path to the analyzed file
            analysis_type: Type of analysis performed
            
        Returns:
            Analysis result or None if not found
        """
        if not self.enable_intelligent_analysis:
            return None
        
        try:
            # Check persistence layer first
            result = self.retrieve_analysis_result(file_path, analysis_type)
            if result is not None:
                return result
            
            # Check in-memory cache
            if analysis_type == 'static' and file_path in self.analysis_cache:
                return self.analysis_cache[file_path]
            elif analysis_type == 'structure' and file_path in self.structure_cache:
                return self.structure_cache[file_path]
            elif analysis_type == 'dependencies' and file_path in self.dependency_cache:
                return self.dependency_cache[file_path]
            elif analysis_type == 'relationships' and file_path in self.relationship_cache:
                return self.relationship_cache[file_path]
            
            return None
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to get cached analysis: {e}")
            return None
    
    def cache_analysis_result(self, file_path, analysis_type, result, persist=True, ttl_hours=None):
        """Cache analysis result in memory and optionally persist it.
        
        Args:
            file_path: Path to the analyzed file
            analysis_type: Type of analysis performed
            result: Analysis result data
            persist: Whether to persist the result
            ttl_hours: Time-to-live in hours (optional)
            
        Returns:
            Success status
        """
        if not self.enable_intelligent_analysis:
            return False
        
        try:
            # Cache in memory
            if analysis_type == 'static':
                self.analysis_cache[file_path] = result
            elif analysis_type == 'structure':
                self.structure_cache[file_path] = result
            elif analysis_type == 'dependencies':
                self.dependency_cache[file_path] = result
            elif analysis_type == 'relationships':
                self.relationship_cache[file_path] = result
            
            # Persist if requested
            if persist:
                success = self.store_analysis_result(
                    file_path=file_path,
                    analysis_type=analysis_type,
                    result=result,
                    ttl_hours=ttl_hours
                )
                return success
            
            return True
            
        except Exception as e:
            if self.verbose:
                self.io.tool_warning(f"Failed to cache analysis result: {e}")
            return False


def find_src_files(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    return src_files


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


def get_scm_fname(lang):
    # Load the tags queries
    if USING_TSL_PACK:
        subdir = "tree-sitter-language-pack"
        try:
            path = resources.files(__package__).joinpath(
                "queries",
                subdir,
                f"{lang}-tags.scm",
            )
            if path.exists():
                return path
        except KeyError:
            pass

    # Fall back to tree-sitter-languages
    subdir = "tree-sitter-languages"
    try:
        return resources.files(__package__).joinpath(
            "queries",
            subdir,
            f"{lang}-tags.scm",
        )
    except KeyError:
        return


def get_supported_languages_md():
    from grep_ast.parsers import PARSERS

    res = """
| Language | File extension | Repo map | Linter |
|:--------:|:--------------:|:--------:|:------:|
"""
    data = sorted((lang, ex) for ex, lang in PARSERS.items())

    for lang, ext in data:
        fn = get_scm_fname(lang)
        repo_map = "✓" if Path(fn).exists() else ""
        linter_support = "✓"
        res += f"| {lang:20} | {ext:20} | {repo_map:^8} | {linter_support:^6} |\n"

    res += "\n"

    return res


if __name__ == "__main__":
    fnames = sys.argv[1:]

    chat_fnames = []
    other_fnames = []
    for fname in sys.argv[1:]:
        if Path(fname).is_dir():
            chat_fnames += find_src_files(fname)
        else:
            chat_fnames.append(fname)

    rm = RepoMap(root=".")
    repo_map = rm.get_ranked_tags_map(chat_fnames, other_fnames)

    dump(len(repo_map))
    print(repo_map)
