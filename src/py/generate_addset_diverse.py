import argparse
import collections
import json
import math
from nectar import corenlp
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize
import os
from pattern import en as patten
import random 
import re
from termcolor import colored
import sys
import numpy as np
import copy

OPTS = None

class ConversionRule(object):
  def convert(self, q, a, tokens, const_parse, run_fix_style=True):
    raise NotImplementedError

class ConstituencyRule(ConversionRule):
  """A rule for converting question to sentence based on constituency parse."""
  def __init__(self, in_pattern, out_pattern, postproc=None):
    self.in_pattern = in_pattern   # e.g. "where did $NP $VP"
    self.out_pattern = unicode(out_pattern)
        # e.g. "{1} did {2} at {0}."  Answer is always 0
    self.name = in_pattern
    if postproc:
      self.postproc = postproc
    else:
      self.postproc = {}

  def convert(self, q, a, tokens, const_parse, run_fix_style=True):
    pattern_toks = self.in_pattern.split(' ')   # Don't care about trailing punctuation
    match = match_pattern(self.in_pattern, const_parse)
    appended_clause = False
    if not match:
      # Try adding a PP at the beginning
      appended_clause = True
      new_pattern = '$PP , ' + self.in_pattern
      pattern_toks = new_pattern.split(' ')
      match = match_pattern(new_pattern, const_parse)
    if not match:
      # Try adding an SBAR at the beginning
      new_pattern = '$SBAR , ' + self.in_pattern
      pattern_toks = new_pattern.split(' ')
      match = match_pattern(new_pattern, const_parse)
    if not match: return None
    appended_clause_match = None
    fmt_args = [a]
    for t, m in zip(pattern_toks, match):
      if t.startswith('$') or '/' in t:
        # First check if it's a WHP
        phrase = convert_whp(m, q, a, tokens)
        if not phrase:
          phrase = m.get_phrase()
        fmt_args.append(phrase)
    if appended_clause:
      appended_clause_match = fmt_args[1]
      fmt_args = [a] + fmt_args[2:]
    for i in range(len(fmt_args)):
      if i in self.postproc:
        # Run postprocessing filters
        fmt_args[i] = run_postprocessing(fmt_args[i], self.postproc[i], fmt_args)
    output = self.gen_output(fmt_args)
    if appended_clause:
      output = appended_clause_match + ', ' + output
    if run_fix_style:
      output = fix_style(output)
    return output

  def gen_output(self, fmt_args):
    """By default, use self.out_pattern.  Can be overridden."""
    return self.out_pattern.format(*fmt_args)

class ReplaceRule(ConversionRule):
  """A simple rule that replaces some tokens with the answer."""
  def __init__(self, target, replacement='{}', start=False):
    self.target = target
    self.replacement = unicode(replacement)
    self.name = 'replace(%s)' % target
    self.start = start

  def convert(self, q, a, tokens, const_parse, run_fix_style=True):
    t_toks = self.target.split(' ')
    q_toks = q.rstrip('?.').split(' ')
    replacement_text = self.replacement.format(a)
    for i in range(len(q_toks)):
      if self.start and i != 0: continue
      if ' '.join(q_toks[i:i + len(t_toks)]).rstrip(',').lower() == self.target:
        begin = q_toks[:i]
        end = q_toks[i + len(t_toks):]
        output = ' '.join(begin + [replacement_text] + end)
        if run_fix_style:
          output = fix_style(output)
        return output
    return None

class FindWHPRule(ConversionRule):
  """A rule that looks for $WHP's from right to left and does replacements."""
  name = 'FindWHP'
  def _recursive_convert(self, node, q, a, tokens, found_whp):
    if node.word: return node.word, found_whp
    if not found_whp:
      whp_phrase = convert_whp(node, q, a, tokens)
      if whp_phrase: return whp_phrase, True
    child_phrases = []
    for c in node.children[::-1]:
      c_phrase, found_whp = self._recursive_convert(c, q, a, tokens, found_whp)
      child_phrases.append(c_phrase)
    out_toks = []
    for i, p in enumerate(child_phrases[::-1]):
      if i == 0 or p.startswith("'"):
        out_toks.append(p)
      else:
        out_toks.append(' ' + p)
    return ''.join(out_toks), found_whp

  def convert(self, q, a, tokens, const_parse, run_fix_style=True):
    out_phrase, found_whp = self._recursive_convert(const_parse, q, a, tokens, False)
    if found_whp:
      if run_fix_style:
        out_phrase = fix_style(out_phrase)
      return out_phrase
    return None

class AnswerRule(ConversionRule):
  """Just return the answer."""
  name = 'AnswerRule'
  def convert(self, q, a, tokens, const_parse, run_fix_style=True):
    return a

CONVERSION_RULES = [
    # Special rules
    ConstituencyRule('$WHP:what $Be $NP called that $VP', '{2} that {3} {1} called {1}'),

    # What type of X
    #ConstituencyRule("$WHP:what/which type/sort/kind/group of $NP/$Noun $Be $NP", '{5} {4} a {1} {3}'),
    #ConstituencyRule("$WHP:what/which type/sort/kind/group of $NP/$Noun $Be $VP", '{1} {3} {4} {5}'),
    #ConstituencyRule("$WHP:what/which type/sort/kind/group of $NP $VP", '{1} {3} {4}'),

    # How $JJ
    ConstituencyRule('how $JJ $Be $NP $IN $NP', '{3} {2} {0} {1} {4} {5}'),
    ConstituencyRule('how $JJ $Be $NP $SBAR', '{3} {2} {0} {1} {4}'),
    ConstituencyRule('how $JJ $Be $NP', '{3} {2} {0} {1}'),

    # When/where $Verb
    ConstituencyRule('$WHP:when/where $Do $NP', '{3} occurred in {1}'),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb', '{3} {4} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP/$PP', '{3} {4} {5} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP $PP', '{3} {4} {5} {6} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Be $NP', '{3} {2} in {1}'),
    ConstituencyRule('$WHP:when/where $Verb $NP $VP/$ADJP', '{3} {2} {4} in {1}'),

    # What/who/how $Do
    ConstituencyRule("$WHP:what/which/who $Do $NP do", '{3} {1}', {0: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb", '{3} {4} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $IN/$NP", '{3} {4} {5} {1}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $PP", '{3} {4} {1} {5}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $NP $VP", '{3} {4} {5} {6} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB", '{3} {4} to {5} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB $VP", '{3} {4} to {5} {1} {6}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $NP $IN $VP", '{3} {4} {5} {6} {1} {7}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP/$S/$VP/$SBAR/$SQ", '{3} {4} {1} {5}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP $PP/$S/$VP/$SBAR", '{3} {4} {1} {5} {6}', {4: 'tense-2'}),

    # What/who/how $Be
    # Watch out for things that end in a preposition
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP of $NP $Verb/$Part $IN", '{3} of {4} {2} {5} {6} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $NP $IN", '{3} {2} {4} {5} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $VP/$IN", '{3} {2} {4} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $IN $NP/$VP", '{1} {2} {3} {4} {5}'),
    ConstituencyRule('$WHP:what/which/who $Be/$MD $NP $Verb $PP', '{3} {2} {4} {1} {5}'),
    ConstituencyRule('$WHP:what/which/who $Be/$MD $NP/$VP/$PP', '{1} {2} {3}'),
    ConstituencyRule("$WHP:how $Be/$MD $NP $VP", '{3} {2} {4} by {1}'),

    # What/who $Verb
    ConstituencyRule("$WHP:what/which/who $VP", '{1} {2}'),

    # $IN what/which $NP
    ConstituencyRule('$IN what/which $NP $Do $NP $Verb $NP', '{5} {6} {7} {1} the {3} of {0}',
                     {1: 'lower', 6: 'tense-4'}),
    ConstituencyRule('$IN what/which $NP $Be $NP $VP/$ADJP', '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    ConstituencyRule('$IN what/which $NP $Verb $NP/$ADJP $VP', '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    FindWHPRule(),
]

# Rules for going from WHP to an answer constituent
WHP_RULES = [
    # WHPP rules
    ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun', '{1} {0} {4}'),
    ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun $PP', '{1} {0} {4} {5}'),
    ConstituencyRule('$IN what/which $NP', '{1} the {3} of {0}'),
    ConstituencyRule('$IN $WP/$WDT', '{1} {0}'),

    # what/which
    ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun', '{0} {3}'),
    ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun $PP', '{0} {3} {4}'),
    ConstituencyRule('what/which $NP', 'the {2} of {0}'),

    # How many
    ConstituencyRule('how many/much $NP', '{0} {2}'),

    # Replace
    ReplaceRule('what'),
    ReplaceRule('who'),
    ReplaceRule('how many'),
    ReplaceRule('how much'),
    ReplaceRule('which'),
    ReplaceRule('where'),
    ReplaceRule('when'),
    ReplaceRule('why'),
    ReplaceRule('how'),

    # Just give the answer
    AnswerRule(),
]

STEMMER = LancasterStemmer()

POS_TO_WORDNET = {
    'NN': wn.NOUN,
    'JJ': wn.ADJ,
    'JJR': wn.ADJ,
    'JJS': wn.ADJ,
}

# Map to pattern.en aliases
# http://www.clips.ua.ac.be/pages/pattern-en#conjugation
POS_TO_PATTERN = {
    'vb': 'inf',  # Infinitive
    'vbp': '1sg',  # non-3rd-person singular present
    'vbz': '3sg',  # 3rd-person singular present
    'vbg': 'part',  # gerund or present participle
    'vbd': 'p',  # past
    'vbn': 'ppart',  # past participle
}
# Tenses prioritized by likelihood of arising
PATTERN_TENSES = ['inf', '3sg', 'p', 'part', 'ppart', '1sg']

# Constants
DATASETS = {
    'dev': 'data/squad/dev-v1.1.json',
    'sample1k': 'out/none_n1000_k1_s0.json',
    'train': 'data/squad/train-v1.1.json',
}
CORENLP_CACHES = {
    'dev': 'data/squad/corenlp_cache.json',
    'sample1k': 'data/squad/corenlp_cache.json',
    'train': 'data/squad/train_corenlp_cache.json',
}
NEARBY_GLOVE_FILE = 'out/nearby_n100_glove_6B_100d.json'
POSTAG_FILE = 'data/postag_dict.json'

CONST_PARSE_MACROS = {
    '$Noun': '$NP/$NN/$NNS/$NNP/$NNPS',
    '$Verb': '$VB/$VBD/$VBP/$VBZ',
    '$Part': '$VBN/$VG',
    '$Be': 'is/are/was/were',
    '$Do': "do/did/does/don't/didn't/doesn't",
    '$WHP': '$WHADJP/$WHADVP/$WHNP/$WHPP',
}

### Rules for converting questions into declarative sentences
def fix_style(s):
  """Minor, general style fixes for questions."""
  s = s.replace('?', '')  # Delete question marks anywhere in sentence.
  s = s.strip(' .')
  if s[0] == s[0].lower():
    s = s[0].upper() + s[1:]
  return s + '.'

def run_postprocessing(s, rules, all_args):
  rule_list = rules.split(',')
  for rule in rule_list:
    if rule == 'lower':
      s = s.lower()
    elif rule.startswith('tense-'):
      ind = int(rule[6:])
      orig_vb = all_args[ind]
      tenses = patten.tenses(orig_vb)
      for tense in PATTERN_TENSES:  # Prioritize by PATTERN_TENSES
        if tense in tenses:
          break
      else:  # Default to first tense
        tense = PATTERN_TENSES[0]
      s = patten.conjugate(s, tense)
    elif rule in POS_TO_PATTERN:
      s = patten.conjugate(s, POS_TO_PATTERN[rule])
  return s

def convert_whp(node, q, a, tokens):
  if node.tag in ('WHNP', 'WHADJP', 'WHADVP', 'WHPP'):
    # Apply WHP rules
    cur_phrase = node.get_phrase()
    cur_tokens = tokens[node.get_start_index():node.get_end_index()]
    for r in WHP_RULES:
      phrase = r.convert(cur_phrase, a, cur_tokens, node, run_fix_style=False)
      if phrase:
        # if not OPTS.quiet:
        #   print ('  WHP Rule "%s": %s' % (r.name, colored(phrase, 'yellow'))).encode('utf-8')
        return phrase
  return None

def _check_match(node, pattern_tok):
  if pattern_tok in CONST_PARSE_MACROS:
    pattern_tok = CONST_PARSE_MACROS[pattern_tok]
  if ':' in pattern_tok:
    # ':' means you match the LHS category and start with something on the right
    lhs, rhs = pattern_tok.split(':')
    match_lhs = _check_match(node, lhs)
    if not match_lhs: return False
    phrase = node.get_phrase().lower()
    retval = any(phrase.startswith(w) for w in rhs.split('/'))
    return retval
  elif '/' in pattern_tok:
    return any(_check_match(node, t) for t in pattern_tok.split('/'))
  return ((pattern_tok.startswith('$') and pattern_tok[1:] == node.tag) or
          (node.word and pattern_tok.lower() == node.word.lower()))

def _recursive_match_pattern(pattern_toks, stack, matches):
  """Recursively try to match a pattern, greedily."""
  if len(matches) == len(pattern_toks):
    # We matched everything in the pattern; also need stack to be empty
    return len(stack) == 0
  if len(stack) == 0: return False
  cur_tok = pattern_toks[len(matches)]
  node = stack.pop()
  # See if we match the current token at this level
  is_match = _check_match(node, cur_tok)
  if is_match:
    cur_num_matches = len(matches)
    matches.append(node)
    new_stack = list(stack)
    success = _recursive_match_pattern(pattern_toks, new_stack, matches)
    if success: return True
    # Backtrack
    while len(matches) > cur_num_matches:
      matches.pop()
  # Recurse to children
  if not node.children: return False  # No children to recurse on, we failed
  stack.extend(node.children[::-1])  # Leftmost children should be popped first
  return _recursive_match_pattern(pattern_toks, stack, matches)

def match_pattern(pattern, const_parse):
  pattern_toks = pattern.split(' ')
  whole_phrase = const_parse.get_phrase()
  if whole_phrase.endswith('?') or whole_phrase.endswith('.'):
    # Match trailing punctuation as needed
    pattern_toks.append(whole_phrase[-1])
  matches = []
  success = _recursive_match_pattern(pattern_toks, [const_parse], matches)
  if success:
    return matches
  else:
    return None

def alter_special(token, **kwargs):
  w = token['originalText']
  if w in SPECIAL_ALTERATIONS:
    return [SPECIAL_ALTERATIONS[w]]
  return None

def alter_nearby(pos_list, ignore_pos=False, is_ner=False):
  def func(token, nearby_word_dict=None, postag_dict=None, **kwargs):
    if token['pos'] not in pos_list: return None
    if is_ner and token['ner'] not in ('PERSON', 'LOCATION', 'ORGANIZATION', 'MISC'):
      return None
    w = token['word'].lower()
    if w in ('war'): return None
    if w not in nearby_word_dict: return None
    new_words = []
    w_stem = STEMMER.stem(w.replace('.', ''))
    for x in nearby_word_dict[w][1:]:
      new_word = x['word']
      # Make sure words aren't too similar (e.g. same stem)
      new_stem = STEMMER.stem(new_word.replace('.', ''))
      if w_stem.startswith(new_stem) or new_stem.startswith(w_stem): continue
      if not ignore_pos:
        # Check for POS tag match
        if new_word not in postag_dict: continue
        new_postag = postag_dict[new_word]
        if new_postag != token['pos']: continue 
      new_words.append(new_word)
    return new_words
  return func

def alter_entity_glove(token, nearby_word_dict=None, **kwargs):
  # NOTE: Deprecated
  if token['ner'] not in ('PERSON', 'LOCATION', 'ORGANIZATION', 'MISC'): return None
  w = token['word'].lower()
  if w == token['word']: return None  # Only do capitalized words
  if w not in nearby_word_dict: return None
  new_words = []
  for x in nearby_word_dict[w][1:3]:
    if token['word'] == w.upper():
      new_words.append(x['word'].upper())
    else:
      new_words.append(x['word'].title())
  return new_words

def alter_entity_type(token, **kwargs):
  pos = token['pos']
  ner = token['ner']
  word = token['word']
  is_abbrev = word == word.upper() and not word == word.lower()
  if token['pos'] not in (
      'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS',
      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
    # Don't alter non-content words
    return None
  if ner == 'PERSON':
    return ['Jackson']
  elif ner == 'LOCATION':
    return ['Berlin']
  elif ner == 'ORGANIZATION':
    if is_abbrev: return ['UNICEF']
    return ['Acme']
  elif ner == 'MISC':
    return ['Neptune']
  elif ner == 'NNP':
    if is_abbrev: return ['XKCD']
    return ['Dalek']
  elif pos == 'NNPS':
    return ['Daleks']
  return None

def alter_wordnet_antonyms(token, **kwargs):
  if token['pos'] not in POS_TO_WORDNET: return None
  w = token['word'].lower()
  wn_pos = POS_TO_WORDNET[token['pos']]
  synsets = wn.synsets(w, wn_pos)
  if not synsets: return None
  synset = synsets[0]
  antonyms = []
  for lem in synset.lemmas():
    if lem.antonyms():
      for a in lem.antonyms():
        new_word = a.name()
        if '_' in a.name(): continue
        antonyms.append(new_word)
  return antonyms


HIGH_CONF_ALTER_RULES = collections.OrderedDict([
    ('special', alter_special),
    ('wn_antonyms', alter_wordnet_antonyms),
    ('nearbyNum', alter_nearby(['CD'], ignore_pos=True)),
    ('nearbyProperNoun', alter_nearby(['NNP', 'NNPS'])),
    ('nearbyProperNoun', alter_nearby(['NNP', 'NNPS'], ignore_pos=True)),
    ('nearbyEntityNouns', alter_nearby(['NN', 'NNS'], is_ner=True)),
    ('nearbyEntityJJ', alter_nearby(['JJ', 'JJR', 'JJS'], is_ner=True)),
    ('entityType', alter_entity_type),
    #('entity_glove', alter_entity_glove),
])

SPECIAL_ALTERATIONS = {
    'States': 'Kingdom',
    'US': 'UK',
    'U.S': 'U.K.',
    'U.S.': 'U.K.',
    'UK': 'US',
    'U.K.': 'U.S.',
    'U.K': 'U.S.',
    'largest': 'smallest',
    'smallest': 'largest',
    'highest': 'lowest',
    'lowest': 'highest',
    'May': 'April',
    'Peyton': 'Trevor',
}

ALL_ALTER_RULES = collections.OrderedDict(HIGH_CONF_ALTER_RULES.items() + [
    ('nearbyAdj', alter_nearby(['JJ', 'JJR', 'JJS'])),
    ('nearbyNoun', alter_nearby(['NN', 'NNS'])),
    #('nearbyNoun', alter_nearby(['NN', 'NNS'], ignore_pos=True)),
])

DO_NOT_ALTER = ['many', 'such', 'few', 'much', 'other', 'same', 'general',
                'type', 'record', 'kind', 'sort', 'part', 'form', 'terms', 'use',
                'place', 'way', 'old', 'young', 'bowl', 'united', 'one',
                'likely', 'different', 'square', 'war', 'republic', 'doctor', 'color']
BAD_ALTERATIONS = ['mx2004', 'planet', 'u.s.', 'Http://Www.Co.Mo.Md.Us']

def ans_number(a, tokens, q, **kwargs):
  out_toks = []
  seen_num = False
  for t in tokens:
    ner = t['ner']
    pos = t['pos']
    w = t['word']
    out_tok = {'before': t['before']}

    # Split on dashes
    leftover = ''
    dash_toks = w.split('-')
    if len(dash_toks) > 1:
      w = dash_toks[0]
      leftover = '-'.join(dash_toks[1:])

    # Try to get a number out
    value = None
    if w != '%': 
      # Percent sign should just pass through
      try:
        value = float(w.replace(',', ''))
      except:
        try:
          norm_ner = t['normalizedNER']
          if norm_ner[0] in ('%', '>', '<'):
            norm_ner = norm_ner[1:]
          value = float(norm_ner)
        except:
          pass
    if not value and (
        ner == 'NUMBER' or 
        (ner == 'PERCENT' and pos == 'CD')):
      # Force this to be a number anyways
      value = 10
    if value:
      if math.isinf(value) or math.isnan(value): value = 9001
      seen_num = True
      if w in ('thousand', 'million', 'billion', 'trillion'):
        if w == 'thousand':
          new_val = 'million'
        else:
          new_val = 'thousand'
      else:
        if value < 2500 and value > 1000:
          new_val = str(value - 75)
        else:
          # Change leading digit
          if value == int(value):
            val_chars = list('%d' % value)
          else:
            val_chars = list('%g' % value)
          c = val_chars[0]
          for i in range(len(val_chars)):
            c = val_chars[i]
            if c >= '0' and c <= '9':
              val_chars[i] = str(max((int(c) + 5) % 10, 1))
              break
          new_val = ''.join(val_chars)
      if leftover:
        new_val = '%s-%s' % (new_val, leftover)
      out_tok['originalText'] = new_val
    else:
      out_tok['originalText'] = t['originalText']
    out_toks.append(out_tok)
  if seen_num:
    return corenlp.rejoin(out_toks).strip()
  else:
    return None

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
          'august', 'september', 'october', 'november', 'december']

def ans_date(a, tokens, q, **kwargs):
  out_toks = []
  if not all(t['ner'] == 'DATE' for t in tokens): return None
  for t in tokens:
    if t['pos'] == 'CD' or t['word'].isdigit():
      try:
        value = int(t['word'])
      except:
        value = 10  # fallback
      if value > 50:  new_val = str(value - 25)  # Year
      else:  # Day of month
        if value > 15: new_val = str(value - 11)
        else: new_val = str(value + 11)
    else:
      if t['word'].lower() in MONTHS:
        m_ind = MONTHS.index(t['word'].lower())
        new_val = MONTHS[(m_ind + 6) % 12].title()
      else:
        # Give up
        new_val = t['originalText']
    out_toks.append({'before': t['before'], 'originalText': new_val})
  new_ans = corenlp.rejoin(out_toks).strip()
  if new_ans == a['text']: return None
  return new_ans

def ans_entity_full(ner_tag, new_ans):
  """Returns a function that yields new_ans iff every token has |ner_tag|."""
  def func(a, tokens, q, **kwargs):
    for t in tokens:
      if t['ner'] != ner_tag: return None
    return new_ans
  return func

def ans_abbrev(new_ans):
  def func(a, tokens, q, **kwargs):
    s = a['text']
    if s == s.upper() and s != s.lower():
      return new_ans
    return None
  return func

def ans_match_wh(wh_word, new_ans):
  """Returns a function that yields new_ans if the question starts with |wh_word|."""
  def func(a, tokens, q, **kwargs):
    if q.lower().startswith(wh_word + ' '):
      return new_ans
    return None
  return func

def ans_pos(pos, new_ans, end=False, add_dt=False):
  """Returns a function that yields new_ans if the first/last token has |pos|."""
  def func(a, tokens, q, determiner, **kwargs):
    if end:
      t = tokens[-1]
    else:
      t = tokens[0]
    if t['pos'] != pos: return None
    if add_dt and determiner:
      return '%s %s' % (determiner, new_ans)
    return new_ans
  return func

def compress_whnp(tree, inside_whnp=False):
  if not tree.children: return tree  # Reached leaf
  # Compress all children
  for i, c in enumerate(tree.children):
    tree.children[i] = compress_whnp(c, inside_whnp=inside_whnp or tree.tag == 'WHNP')
  if tree.tag != 'WHNP':
    if inside_whnp:
      # Wrap everything in an NP
      return corenlp.ConstituencyParse('NP', children=[tree])
    return tree
  wh_word = None
  new_np_children = []
  new_siblings = []
  for i, c in enumerate(tree.children):
    if i == 0:
      if c.tag in ('WHNP', 'WHADJP', 'WHAVP', 'WHPP'):
        wh_word = c.children[0]
        new_np_children.extend(c.children[1:])
      elif c.tag in ('WDT', 'WP', 'WP$', 'WRB'):
        wh_word = c
      else:
        # No WH-word at start of WHNP
        return tree
    else:
      if c.tag == 'SQ':  # Due to bad parse, SQ may show up here
        new_siblings = tree.children[i:]
        break
      # Wrap everything in an NP
      new_np_children.append(corenlp.ConstituencyParse('NP', children=[c]))
  if new_np_children:
    new_np = corenlp.ConstituencyParse('NP', children=new_np_children)
    new_tree = corenlp.ConstituencyParse('WHNP', children=[wh_word, new_np])
  else:
    new_tree = tree
  if new_siblings:
    new_tree = corenlp.ConstituencyParse('SBARQ', children=[new_tree] + new_siblings)
  return new_tree

def read_const_parse(parse_str):
  tree = corenlp.ConstituencyParse.from_corenlp(parse_str)
  new_tree = compress_whnp(tree)
  return new_tree
  
def ans_catch_all(new_ans):
  def func(a, tokens, q, **kwargs):
    return new_ans
  return func

ANSWER_RULES = [
    ('date', ans_date),
    ('number', ans_number),
    ('ner_person', ans_entity_full('PERSON', 'Jeff Dean')),
    ('ner_location', ans_entity_full('LOCATION', 'Chicago')),
    ('ner_organization', ans_entity_full('ORGANIZATION', 'Stark Industries')),
    ('ner_misc', ans_entity_full('MISC', 'Jupiter')),
    ('abbrev', ans_abbrev('LSTM')),
    ('wh_who', ans_match_wh('who', 'Jeff Dean')),
    ('wh_when', ans_match_wh('when', '1956')),
    ('wh_where', ans_match_wh('where', 'Chicago')),
    ('wh_where', ans_match_wh('how many', '42')),
    # Starts with verb
    ('pos_begin_vb', ans_pos('VB', 'learn')),
    ('pos_end_vbd', ans_pos('VBD', 'learned')),
    ('pos_end_vbg', ans_pos('VBG', 'learning')),
    ('pos_end_vbp', ans_pos('VBP', 'learns')),
    ('pos_end_vbz', ans_pos('VBZ', 'learns')),
    # Ends with some POS tag
    ('pos_end_nn', ans_pos('NN', 'hamster', end=True, add_dt=True)),
    ('pos_end_nnp', ans_pos('NNP', 'Central Park', end=True, add_dt=True)),
    ('pos_end_nns', ans_pos('NNS', 'hamsters', end=True, add_dt=True)),
    ('pos_end_nnps', ans_pos('NNPS', 'Kew Gardens', end=True, add_dt=True)),
    ('pos_end_jj', ans_pos('JJ', 'deep', end=True)),
    ('pos_end_jjr', ans_pos('JJR', 'deeper', end=True)),
    ('pos_end_jjs', ans_pos('JJS', 'deepest', end=True)),
    ('pos_end_rb', ans_pos('RB', 'silently', end=True)),
    ('pos_end_vbg', ans_pos('VBG', 'learning', end=True)),
    ('catch_all', ans_catch_all('aliens')),
]

def read_data(filename):
  with open(filename) as f:
    return json.load(f)

def load_cache():
  cache_file = CORENLP_CACHES[OPTS.dataset]
  with open(cache_file) as f:
    return json.load(f)

def load_postag_dict():
  with open(POSTAG_FILE) as f:
    return json.load(f)

def load_nearby_words():
  with open(NEARBY_GLOVE_FILE) as f:
    return json.load(f)

def load_sentence_dict():
  # TODO find solution for dynamic path
  sentence_dict = OPTS.sentence_dict
  with open(sentence_dict) as f:
    return json.load(f)

class AddSentDiverseGenerator:

  def __init__(self):
    self.load_resources()

  def load_resources(self):
    print("loading CORENLP Cache")
    self.corenlp_cache = load_cache()
    print("loading Glove Features")
    self.nearby_word_dict = load_nearby_words()
    print("loading Postag Dict")
    self.postag_dict = load_postag_dict()
    print("loading sentence dict")
    self.sentence_dict = load_sentence_dict()

  def alter_question(self, q, tokens, const_parse,
                    strategy='separate'):
    """Alter the question to make it ask something else.

    Possible strategies:
      - separate: Do best alteration for each word separately.
      - best: Generate exactly one best alteration (may over-alter).
      - high-conf: Do all possible high-confidence alterations
      - high-conf-separate: Do best high-confidence alteration for each word separately.
      - all: Do all possible alterations (very conservative)
    """
    used_words = [t['word'].lower() for t in tokens]
    new_qs = []
    toks_all = []
    if strategy.startswith('high-conf'): 
      rules = HIGH_CONF_ALTER_RULES
    else:
      rules = ALL_ALTER_RULES
    for i, t in enumerate(tokens):
      if t['word'].lower() in DO_NOT_ALTER: 
        if strategy in ('high-conf', 'all'): toks_all.append(t)
        continue
      begin = tokens[:i]
      end = tokens[i+1:]
      found = False
      for rule_name in rules:
        rule = rules[rule_name]
        new_words = rule(t, nearby_word_dict=self.nearby_word_dict,
                        postag_dict=self.postag_dict)
        if new_words:
          for nw in new_words:
            if nw.lower() in used_words: continue
            if nw.lower() in BAD_ALTERATIONS: continue
            # Match capitzliation
            if t['word'] == t['word'].upper():
              nw = nw.upper()
            elif t['word'] == t['word'].title():
              nw = nw.title()
            new_tok = dict(t)
            new_tok['word'] = new_tok['lemma'] = new_tok['originalText'] = nw
            new_tok['altered'] = True
            # NOTE: obviously this is approximate
            if strategy.endswith('separate'):
              new_tokens = begin + [new_tok] + end
              new_q = corenlp.rejoin(new_tokens)
              tag = '%s-%d-%s' % (rule_name, i, nw)
              new_const_parse = corenlp.ConstituencyParse.replace_words(
                  const_parse, [t['word'] for t in new_tokens])
              new_qs.append((new_q, new_tokens, new_const_parse, tag))
              break
            elif strategy in ('high-conf', 'all'):
              toks_all.append(new_tok)
              found = True
              break
        if strategy in ('high-conf', 'all') and found: break
      if strategy in ('high-conf', 'all') and not found:
        toks_all.append(t)
    if strategy in ('high-conf', 'all'):
      new_q = corenlp.rejoin(toks_all)
      new_const_parse = corenlp.ConstituencyParse.replace_words(
          const_parse, [t['word'] for t in toks_all])
      if new_q != q:
        new_qs.append((corenlp.rejoin(toks_all), toks_all, new_const_parse, strategy))
    return new_qs

  def mutate_question(self, question_text, alteration_strategy='high-conf'):

    # get tokens of pared question
    q_parse = self.corenlp_cache[question_text]
    q_tokens = q_parse['tokens']
    q_const_parse = read_const_parse(q_parse['parse'])

    # Mutate Question aligned to strategy
    if alteration_strategy:
      # Easiest to alter the question before converting
      q_list = self.alter_question(
          question_text, q_tokens, q_const_parse, strategy=alteration_strategy)
    else:
      q_list = [(question, q_tokens, q_const_parse, 'unaltered')]

    return q_list

  def merge_question_answer(self, question_list, answer_text):

    # bring mutated question and fake answer together to one declarative sentence
    for q_str, q_tokens, q_const_parse, tag in question_list:
      for rule in CONVERSION_RULES:
        sent = rule.convert(q_str, answer_text, q_tokens, q_const_parse)
        
        if sent:
          return sent

  def _get_tokens_for_answers(self, answer_objs, corenlp_obj):
    """Get CoreNLP tokens corresponding to a SQuAD answer object."""
    first_a_toks = None
    for i, a_obj in enumerate(answer_objs):
      a_toks = []
      answer_start = a_obj['answer_start']
      answer_end = answer_start + len(a_obj['text'])
      for s in corenlp_obj['sentences']:
        for t in s['tokens']:
          if t['characterOffsetBegin']  >= answer_end: continue
          if t['characterOffsetEnd'] <= answer_start: continue
          a_toks.append(t)
      if corenlp.rejoin(a_toks).strip() == a_obj['text']:
        # Make sure that the tokens reconstruct the answer
        return i, a_toks
      if i == 0: first_a_toks = a_toks
    # None of the extracted token lists reconstruct the answer
    # Default to the first
    return 0, first_a_toks

  def _get_determiner_for_answers(self, answer_objs):
    for a in answer_objs:
      words = a['text'].split(' ')
      if words[0].lower() == 'the': return 'the'
      if words[0].lower() in ('a', 'an'): return 'a'
    return None

  def get_answer_type(self, paragraph_obj, qa_obj):

    question = qa_obj['question'].strip()

    # get tokens of parsed paragraph
    p_parse = self.corenlp_cache[paragraph_obj['context']]

    # get answer tokens
    ind, a_toks = self._get_tokens_for_answers(qa_obj['answers'], p_parse)

    # check wheather answer begin with "the" or "a" or None of both
    determiner = self._get_determiner_for_answers(qa_obj['answers'])
    answer_obj = qa_obj['answers'][ind]

    # find and apply a fitting rule to answer object
    for rule_name, func in ANSWER_RULES:
      answer = func(answer_obj, a_toks, question, determiner=determiner)
      if answer: break
    else:
      raise ValueError('Missing answer')
    
    return rule_name, answer

  def get_same_type_answer(self, rule, qa_obj):

    if rule not in self.sentence_dict:
      return None

    # get set of alternatives
    ans_obj_list = self.sentence_dict[rule]

    # get random iteration numeration
    rand_inds = np.random.permutation(len(ans_obj_list))

    # iterate over list
    for i in rand_inds:
      cur_alt = ans_obj_list[i]

      # if alternative answer found with different qas_id - return
      if cur_alt['qas_id'] != qa_obj['id']:
        return cur_alt

    # else return first
    return ans_obj_list[rand_inds[0]]    

  def generate_paragraph(self, paragraph_obj, qa_obj, adv_sentence_text):
    
    # get paragraph text
    paragraph_text = paragraph_obj['context']

    new_qa_obj = copy.deepcopy(qa_obj)

    # separate paragraph text into sentences
    sent_list = sent_tokenize(paragraph_text)

    # randomly select an insertion placement
    rand_pos = random.randint(0, len(sent_list) + 1)

    # insert adversarial text
    sent_list.insert(rand_pos, adv_sentence_text)

    # print('ADV Sentence: {}'.format(adv_sentence_text))
    # print(sent_list)

    # get position of insterted answer
    pre_adv = " ".join(sent_list[0:(rand_pos)])

    # shift original answer position if necessarry
    for a in new_qa_obj['answers']:
      if (len(pre_adv) < a['answer_start']) or (rand_pos == 0):
        a['answer_start'] += len(adv_sentence_text) + 1

    # create QAS object
    new_qa_obj['id'] += "-addsentdiv"
    new_qa_obj['adversarial_start'] = len(pre_adv) + 1

    if rand_pos == 0:
      new_qa_obj['adversarial_start'] = 0

    new_qa_obj['adversarial_text'] = adv_sentence_text
    new_paragraph = {'context': " ".join(sent_list), 'qas': [new_qa_obj]}

    # return new adversarial QAS object
    return new_paragraph

  def generate_sample(self, paragraph_obj, qa_obj):

    question = qa_obj['question'].strip()

    # Mutate Question aligned to strategy
    q_list = self.mutate_question(question)

    # get orig answer type
    rule_name, _ = self.get_answer_type(paragraph_obj, qa_obj)

    # get fake answer with same type
    fake_ans_obj = self.get_same_type_answer(rule_name, qa_obj)

    # bring mutated question and fake answer together to one declarative sentence
    adv_sent = self.merge_question_answer(q_list, fake_ans_obj['text'])

    if adv_sent is None:
      return None

    # insert sentence to paragraph
    new_paragraph = self.generate_paragraph(paragraph_obj, qa_obj, adv_sent)

    return new_paragraph

  def save_to_json(self, out_obj, out_name, indent=True):
    if not indent:
      with open(out_name, 'w') as f:
        json.dump(out_obj, f)
    else:
      with open(out_name, 'w') as f:
        json.dump(out_obj, f, indent=2)

  def generate_dataset(self, dataset_file, out_name):

    # read dataset
    print("reading input dataset")
    dataset = read_data(dataset_file)

    out_data = []
    counter = 0

    print("generating new paragraphs")
    # iterate over all paragraphs
    for article in dataset['data']:
      out_paragraphs = []
      out_article = {'title': article['title'], 'paragraphs': out_paragraphs}
      out_data.append(out_article)
      for paragraph in article['paragraphs']:

        counter += 1
        print("processing paragraph {}/{}".format(counter, len(article['paragraphs'])))

        # append original paragraph
        out_paragraphs.append(paragraph)

        # iterate over all questions
        for qa_obj in paragraph['qas']:
          new_parag = gen.generate_sample(paragraph, qa_obj)

          if new_parag:
            out_paragraphs.append(new_parag)

    print("generated {} adversarial paragraphs".format(len(out_paragraphs)))
    print("saving results to {}".format(out_name))
    self.save_to_json({"data": out_data, "version": "1.1"}, out_name)


def parse_args():
  parser = argparse.ArgumentParser('Creates a dict of all answers in the dataset with their rule type as key')

  parser.add_argument('--dataset', '-d', default='dev',
                      help='Which dataset (options: [%s])' % (', '.join(DATASETS)))

  parser.add_argument('--sentence_dict', '-s', default='dev',
                      help='Filepath of the generated sentence dictionary')

  parser.add_argument('--outfile', '-o', default='sentence_dict.json',
                      help='Filepath of where to save the dictionary.')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

if __name__ == "__main__":

  OPTS = parse_args()
  dataset_file = DATASETS[OPTS.dataset]

  gen = AddSentDiverseGenerator()
  gen.generate_dataset(dataset_file, OPTS.outfile)
