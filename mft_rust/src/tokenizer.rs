use ahash::AHashMap;
use smallvec::SmallVec;
use indexmap::IndexMap;
use crate::decoder::TurkishDecoder;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenType {
    ROOT,
    SUFFIX,
    BPE,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token: String,
    pub id: i32,
    pub token_type: TokenType,
}

/// Trie node for fast prefix matching - using array for common ASCII chars
struct TrieNode {
    children: AHashMap<char, Box<TrieNode>>,
    value: Option<(i32, usize)>, // (id, char_count)
}

impl TrieNode {
    #[inline]
    fn new() -> Self {
        Self {
            children: AHashMap::with_capacity(4),
            value: None,
        }
    }
    
    #[inline]
    fn get_child(&self, c: char) -> Option<&TrieNode> {
        self.children.get(&c).map(|b| b.as_ref())
    }
}

/// Trie for O(n) prefix matching
pub struct Trie {
    root: TrieNode,
}

impl Trie {
    fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    fn insert(&mut self, key: &str, value: i32) {
        let mut node = &mut self.root;
        let char_count = key.chars().count();
        for c in key.chars() {
            node = node.children.entry(c).or_insert_with(|| Box::new(TrieNode::new()));
        }
        node.value = Some((value, char_count));
    }

    /// Find all prefix matches - returns in descending length order
    #[inline(always)]
    fn find_all_prefixes<'a>(&self, s: &'a str) -> SmallVec<[(i32, usize, usize); 8]> {
        // Returns (id, byte_len, char_count)
        let mut matches = SmallVec::new();
        let mut node = &self.root;
        let mut byte_pos = 0;

        for (i, c) in s.char_indices() {
            if let Some(child) = node.get_child(c) {
                node = child;
                byte_pos = i + c.len_utf8();
                if let Some((id, char_count)) = node.value {
                    matches.push((id, byte_pos, char_count));
                }
            } else {
                break;
            }
        }
        matches.reverse();
        matches
    }

    /// Find longest prefix match only
    #[inline(always)]
    fn find_longest_prefix_info(&self, s: &str) -> Option<(i32, usize, usize)> {
        let mut node = &self.root;
        let mut last_match: Option<(i32, usize, usize)> = None;
        let mut byte_pos = 0;

        for (i, c) in s.char_indices() {
            if let Some(child) = node.get_child(c) {
                node = child;
                byte_pos = i + c.len_utf8();
                if let Some((id, char_count)) = node.value {
                    last_match = Some((id, byte_pos, char_count));
                }
            } else {
                break;
            }
        }
        last_match
    }
}

pub struct TurkishTokenizer {
    roots_trie: Trie,
    suffixes_trie: Trie,
    bpe_trie: Trie,
    reverse_dict: AHashMap<i32, Vec<String>>,
    decoder: TurkishDecoder,
    
    // Special tokens
    uppercase_id: i32,
    unknown_id: i32,
    space_id: i32,
    #[allow(dead_code)]
    pad_id: i32,
    #[allow(dead_code)]
    eos_id: i32,
}

impl TurkishTokenizer {
    pub fn from_files(
        roots_json: &str,
        ekler_json: &str,
        bpe_json: &str,
    ) -> Result<Self, serde_json::Error> {
        let roots: IndexMap<String, i32> = serde_json::from_str(roots_json)?;
        let suffixes: IndexMap<String, i32> = serde_json::from_str(ekler_json)?;
        let bpe_tokens: IndexMap<String, i32> = serde_json::from_str(bpe_json)?;
        
        // Build Tries
        let mut roots_trie = Trie::new();
        let mut suffixes_trie = Trie::new();
        let mut bpe_trie = Trie::new();
        
        for (k, &v) in &roots {
            roots_trie.insert(k, v);
        }
        for (k, &v) in &suffixes {
            suffixes_trie.insert(k, v);
        }
        for (k, &v) in &bpe_tokens {
            bpe_trie.insert(k, v);
        }

        // Build reverse dict with AHashMap
        let mut reverse_dict: AHashMap<i32, Vec<String>> = AHashMap::new();
        let mut insert_rev = |map: &IndexMap<String, i32>| {
            for (k, &v) in map {
                reverse_dict.entry(v).or_default().push(k.clone());
            }
        };
        insert_rev(&roots);
        insert_rev(&suffixes);
        insert_rev(&bpe_tokens);
        
        // Convert to std HashMap for decoder
        let std_reverse: std::collections::HashMap<i32, Vec<String>> = 
            reverse_dict.iter().map(|(&k, v)| (k, v.clone())).collect();
        let decoder = TurkishDecoder::new(std_reverse);
        
        let uppercase_id = *roots.get("<uppercase>").unwrap_or(&0);
        let unknown_id = *roots.get("<unknown>").unwrap_or(&1);
        let space_id = *roots.get(" ").unwrap_or(&2);
        let pad_id = *roots.get("<pad>").unwrap_or(&0);
        let eos_id = *roots.get("<eos>").unwrap_or(&0);

        Ok(Self {
            roots_trie,
            suffixes_trie,
            bpe_trie,
            reverse_dict,
            decoder,
            uppercase_id,
            unknown_id,
            space_id,
            pad_id,
            eos_id,
        })
    }
    
    /// Fast Turkish lowercase - inlined for performance
    #[inline(always)]
    fn tr_lower_char(c: char) -> char {
        match c {
            'İ' => 'i',
            'I' => 'ı',
            'A'..='Z' => ((c as u8) + 32) as char,
            'Ç' => 'ç',
            'Ğ' => 'ğ',
            'Ö' => 'ö',
            'Ş' => 'ş',
            'Ü' => 'ü',
            _ => c,
        }
    }
    
    /// Optimized tokenize for a segment (already lowercased)
    #[inline]
    fn tokenize_segment_fast(&self, s: &str, result: &mut SmallVec<[i32; 64]>) {
        let s_len = s.len();
        let mut pos_byte = 0;
        
        while pos_byte < s_len {
            let substr = &s[pos_byte..];
            
            // Try each trie and collect matches
            let r_matches = self.roots_trie.find_all_prefixes(substr);
            let b_matches = self.bpe_trie.find_all_prefixes(substr);
            let s_matches = self.suffixes_trie.find_all_prefixes(substr);
            
            let mut best_score = 0usize;
            let mut best_priority = 3i32;
            let mut best_byte_len = 0usize;
            let mut best_id = self.unknown_id;
            
            // Score Roots (priority 0 - highest)
            for &(id, byte_len, char_count) in r_matches.iter() {
                let score = char_count * 10;
                if score > best_score || (score == best_score && 0 < best_priority) {
                    best_score = score;
                    best_priority = 0;
                    best_byte_len = byte_len;
                    best_id = id;
                }
            }
            
            // Score BPEs (priority 1) with suffix lookahead
            for &(id, byte_len, char_count) in b_matches.iter() {
                let mut score = char_count * 10;
                
                // Suffix lookahead bonus
                if byte_len < substr.len() {
                    let remainder = &substr[byte_len..];
                    if let Some((_, _, next_char_count)) = self.suffixes_trie.find_longest_prefix_info(remainder) {
                        if next_char_count >= 2 {
                            score += next_char_count;
                        }
                    }
                }
                
                if score > best_score || (score == best_score && 1 < best_priority) {
                    best_score = score;
                    best_priority = 1;
                    best_byte_len = byte_len;
                    best_id = id;
                }
            }
            
            // Score Suffixes (priority 2) with suffix lookahead
            for &(id, byte_len, char_count) in s_matches.iter() {
                let mut score = char_count * 10;
                
                // Suffix lookahead bonus
                if byte_len < substr.len() {
                    let remainder = &substr[byte_len..];
                    if let Some((_, _, next_char_count)) = self.suffixes_trie.find_longest_prefix_info(remainder) {
                        if next_char_count >= 2 {
                            score += next_char_count;
                        }
                    }
                }
                
                if score > best_score || (score == best_score && 2 < best_priority) {
                    best_score = score;
                    best_priority = 2;
                    best_byte_len = byte_len;
                    best_id = id;
                }
            }
            
            if best_priority == 3 {
                // No match found - emit unknown and skip one char
                result.push(self.unknown_id);
                if let Some(c) = substr.chars().next() { 
                    pos_byte += c.len_utf8(); 
                } else { 
                    break; 
                }
            } else {
                result.push(best_id);
                pos_byte += best_byte_len;
            }
        }
    }
    
    /// Fast encode with minimal allocations
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let estimated_tokens = text.len() / 4;
        let mut all_ids: SmallVec<[i32; 64]> = SmallVec::with_capacity(estimated_tokens.min(64));
        
        // Reusable buffer for lowercased text
        let mut lower_buf = String::with_capacity(64);
        
        for part in text.split(' ') {
            let part_trimmed = part.trim();
            if part_trimmed.is_empty() {
                continue;
            }
            
            // Build lowercased version with leading space
            lower_buf.clear();
            lower_buf.push(' ');
            
            let mut has_uppercase = false;
            let mut first_char_uppercase = false;
            let mut first_char_pos = 0;
            
            for (i, c) in part_trimmed.char_indices() {
                let is_upper = c.is_uppercase();
                if is_upper {
                    if i == 0 {
                        first_char_uppercase = true;
                        first_char_pos = lower_buf.len();
                    } else {
                        has_uppercase = true;
                    }
                }
                lower_buf.push(Self::tr_lower_char(c));
            }
            
            if !has_uppercase {
                // Simple case - no internal uppercase
                if first_char_uppercase {
                    all_ids.push(self.uppercase_id);
                }
                self.tokenize_segment_fast(&lower_buf, &mut all_ids);
            } else {
                // Complex case - handle CamelCase
                // For now, fall back to a simpler approach
                if first_char_uppercase {
                    all_ids.push(self.uppercase_id);
                }
                
                let mut last_was_upper = first_char_uppercase;
                let mut segment_start = 1; // Skip leading space
                
                for (i, c) in lower_buf[1..].char_indices() {
                    let orig_idx = i;
                    let orig_c = part_trimmed.chars().nth(orig_idx);
                    
                    if let Some(oc) = orig_c {
                        if oc.is_uppercase() && !last_was_upper && i > 0 {
                            // New uppercase segment - tokenize previous
                            let segment = &lower_buf[segment_start..i+1];
                            self.tokenize_segment_fast(segment, &mut all_ids);
                            all_ids.push(self.uppercase_id);
                            segment_start = i + 1;
                        }
                        last_was_upper = oc.is_uppercase();
                    }
                }
                
                // Tokenize remaining
                if segment_start < lower_buf.len() {
                    let segment = &lower_buf[segment_start..];
                    if !segment.is_empty() {
                        self.tokenize_segment_fast(segment, &mut all_ids);
                    }
                }
            }
        }
        
        // Space removal for uppercase tokens
        let mut final_ids: Vec<i32> = Vec::with_capacity(all_ids.len());
        
        for i in 0..all_ids.len() {
            let id = all_ids[i];
            
            if id == self.uppercase_id && !final_ids.is_empty() {
                if let Some(&last_id) = final_ids.last() {
                    if last_id == self.space_id && i + 1 < all_ids.len() {
                        if let Some(strs) = self.reverse_dict.get(&all_ids[i+1]) {
                            if !strs.is_empty() && strs[0].starts_with(' ') {
                                final_ids.pop();
                            }
                        }
                    }
                }
            }
            final_ids.push(id);
        }
        
        final_ids
    }

    pub fn tokenize_text(&self, text: &str) -> Vec<Token> {
        let ids = self.encode(text);
        ids.iter().map(|&id| {
            let token_str = if let Some(strs) = self.reverse_dict.get(&id) {
                if !strs.is_empty() { strs[0].clone() } else { format!("<id:{}>", id) }
            } else {
                format!("<id:{}>", id)
            };
            
            let token_type = if id < 20000 {
                TokenType::ROOT
            } else if id <= 20071 {
                TokenType::SUFFIX
            } else {
                TokenType::BPE
            };
            
            Token { token: token_str, id, token_type }
        }).collect()
    }
    
    pub fn decode(&self, ids: Vec<i32>) -> String {
        self.decoder.decode(ids)
    }
}
