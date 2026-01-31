use std::collections::HashMap;

pub struct TurkishDecoder {
    reverse_dict: HashMap<i32, Vec<String>>,
}

// Constants for vowel/consonant sets
const ALL_VOWELS: &str = "aeıioöuüâ";
const INCE_VOWELS: &str = "eiöü";
const AI_VOWELS: &str = "aıâ";
const EI_VOWELS: &str = "ei";
const OU_VOWELS: &str = "ou";
const HARD_CONSONANTS: &str = "fstkçşhp";
const WHITESPACE: &str = " \n\t";

impl TurkishDecoder {
    pub fn new(reverse_dict: HashMap<i32, Vec<String>>) -> Self {
        Self { reverse_dict }
    }

    fn has_vowel(s: &str) -> bool {
        s.chars().any(|c| ALL_VOWELS.contains(c))
    }

    fn starts_with_vowel(&self, word: &str) -> bool {
        word.chars().next().map_or(false, |c| ALL_VOWELS.contains(c))
    }

    fn ends_with_vowel(&self, word: &str) -> bool {
        word.chars().last().map_or(false, |c| ALL_VOWELS.contains(c))
    }

    fn ends_with_any(&self, word: &str, charset: &str) -> bool {
        for c in word.chars().rev() {
            if charset.contains(c) {
                return true;
            }
            if ALL_VOWELS.contains(c) {
                return false;
            }
        }
        false
    }
    
    fn ends_with_ince(&self, word: &str) -> bool {
        match word {
            "saat" | "kilovatsaat" | "ziraat" | "itaat" | "istikbal" => true,
            _ => self.ends_with_any(word, INCE_VOWELS),
        }
    }

    fn ends_with_sert_unsuz(&self, word: &str) -> bool {
        word.chars().last().map_or(false, |c| HARD_CONSONANTS.contains(c))
    }
    
    fn get_vowel_suffix_index(&self, prev_token: &str) -> usize {
        if self.ends_with_any(prev_token, AI_VOWELS) {
            0
        } else if self.ends_with_any(prev_token, EI_VOWELS) {
            1
        } else if self.ends_with_any(prev_token, OU_VOWELS) {
            2
        } else {
            3
        }
    }

    fn handle_la_le_suffix(&self, prev_token: &str, suffixes: &[String], end_of_word: bool) -> String {
        if self.ends_with_vowel(prev_token) && end_of_word {
            if self.ends_with_ince(prev_token) {
                suffixes[3].clone() // yle
            } else {
                suffixes[2].clone() // yla
            }
        } else {
            if self.ends_with_ince(prev_token) {
                suffixes[1].clone() // le
            } else {
                suffixes[0].clone() // la
            }
        }
    }

    fn handle_da_de_suffix(&self, prev_token: &str, suffixes: &[String]) -> String {
        if self.ends_with_sert_unsuz(prev_token) {
            if self.ends_with_ince(prev_token) {
                suffixes[3].clone() // te
            } else {
                suffixes[2].clone() // ta
            }
        } else {
            if self.ends_with_ince(prev_token) {
                suffixes[1].clone() // de
            } else {
                suffixes[0].clone() // da
            }
        }
    }

    fn handle_di_du_suffix(&self, prev_token: &str, suffixes: &[String]) -> String {
        let base_index = self.get_vowel_suffix_index(prev_token);
        if self.ends_with_sert_unsuz(prev_token) {
            suffixes[base_index + 4].clone()
        } else {
            suffixes[base_index].clone()
        }
    }
    
    fn handle_lik_suffix(&self, i: usize, ids: &[i32], prev_token: &str, suffixes: &[String]) -> String {
          if i >= ids.len() - 1 {
              return suffixes[0].clone();
          }
          
          let next_token = &self.reverse_dict[&ids[i + 1]][0];
          let base_index = self.get_vowel_suffix_index(prev_token);
          
          if self.starts_with_vowel(next_token) {
              suffixes[base_index + 4].clone()
          } else {
              suffixes[base_index].clone()
          }
    }

    fn handle_cik_suffix(&self, i: usize, ids: &[i32], prev_token: &str, suffixes: &[String]) -> String {
        if i >= ids.len() - 1 {
            return suffixes[0].clone();
        }
        
        let next_token = &self.reverse_dict[&ids[i + 1]][0];
        let base_index = self.get_vowel_suffix_index(prev_token);
        
        let offset = if self.starts_with_vowel(next_token) {
            if self.ends_with_sert_unsuz(prev_token) { 12 } else { 8 }
        } else {
            if self.ends_with_sert_unsuz(prev_token) { 4 } else { 0 }
        };
        
        suffixes[base_index + offset].clone()
    }
    
    fn handle_mak_suffix(&self, i: usize, ids: &[i32], prev_token: &str, suffixes: &[String]) -> String {
        if i >= ids.len() - 1 {
            return suffixes[0].clone();
        }
        
        let next_token = &self.reverse_dict[&ids[i + 1]][0];
        let base_index = if self.ends_with_ince(prev_token) { 1 } else { 0 };
        
        if self.starts_with_vowel(next_token) {
            suffixes[base_index + 2].clone()
        } else {
            suffixes[base_index].clone()
        }
    }
    
    fn handle_acak_suffix(&self, i: usize, ids: &[i32], prev_token: &str, suffixes: &[String]) -> String {
        let is_vowel_ending = self.ends_with_vowel(prev_token);
        let is_ince = self.ends_with_ince(prev_token);
        
        let is_vowel_starting = if i < ids.len() - 1 {
             let next_token = &self.reverse_dict[&ids[i + 1]][0];
             self.starts_with_vowel(next_token)
        } else {
             false
        };
        
        if is_vowel_starting {
            if is_vowel_ending {
                suffixes[if is_ince { 7 } else { 6 }].clone()
            } else {
                 suffixes[if is_ince { 3 } else { 2 }].clone()
            }
        } else {
            if is_vowel_ending {
                 suffixes[if is_ince { 5 } else { 4 }].clone()
            } else {
                 suffixes[if is_ince { 1 } else { 0 }].clone()
            }
        }
    }
    
    fn select_correct_suffix(&self, i: usize, ids: &[i32], prev_token: &str) -> String {
        let token_id = ids[i];
        let suffixes = &self.reverse_dict[&token_id];
        
        if token_id < 20013 {
             if self.ends_with_ince(prev_token) { suffixes[1].clone() } else { suffixes[0].clone() }
        } else if token_id < 20023 {
             suffixes[self.get_vowel_suffix_index(prev_token)].clone()
        } else if token_id == 20023 { // la, le
             let mut end_of_word = true;
             if i < ids.len() - 1 {
                 let next_token = &self.reverse_dict[&ids[i + 1]][0];
                 if !WHITESPACE.contains(next_token.chars().next().unwrap_or(' ')) {
                     end_of_word = false;
                 }
             }
             self.handle_la_le_suffix(prev_token, suffixes, end_of_word)
        } else if token_id <= 20025 { // da, de, tan...
             self.handle_da_de_suffix(prev_token, suffixes)
        } else if token_id < 20029 { // di, du...
             self.handle_di_du_suffix(prev_token, suffixes)
        } else if token_id == 20029 { // lik
             self.handle_lik_suffix(i, ids, prev_token, suffixes)
        } else if token_id == 20030 { // cik
             self.handle_cik_suffix(i, ids, prev_token, suffixes)
        } else if token_id == 20031 { // mak
             self.handle_mak_suffix(i, ids, prev_token, suffixes)
        } else if token_id == 20032 { // acak
             self.handle_acak_suffix(i, ids, prev_token, suffixes)
        } else {
             suffixes[0].clone()
        }
    }

    fn select_correct_root(&self, i: usize, ids: &[i32]) -> String {
        let token_id = ids[i];
        let tokens = &self.reverse_dict[&token_id];
        
        if i >= ids.len() - 1 {
            return tokens[0].clone();
        }
        
        let next_token = &self.reverse_dict[&ids[i + 1]][0];
        
        if token_id == 204 { // "hayat"
             return tokens[0].clone();
        }

        // Meslek Exception (298) - Don't soften to mesleğ
        if token_id == 298 {
             return tokens[0].clone();
        }

        // Akış (aka/akı) Exception (2199)
        if token_id == 2199 {
            if i < ids.len() - 1 {
                 let next_id = ids[i+1];
                 // 32681 = 'cı'
                 // 20080 = 'ş'
                 if next_id == 20080 || next_id == 20100 || next_id == 32681 {
                      return tokens[1].clone();
                 }
                 let next_str = &self.reverse_dict[&next_id][0];
                 if next_str.starts_with('ş') || next_str.starts_with('ı') {
                      return tokens[1].clone();
                 }
            }
        }
        
        // Yaşına (yaşa/yaşı) Exception (2209)
        if token_id == 2209 {
             if i < ids.len() - 1 {
                 // 20188 = 'na'
                 if ids[i+1] == 20188 {
                      return tokens[1].clone();
                 }
             }
        }
        
        // Alın (alın/aln) Exception (182)
        // Default logic drops vowel (100-2080). We want to KEEP vowel 0 for 'ır', 'an'
        if token_id == 182 {
             if i < ids.len() - 1 {
                 let next_id = ids[i+1];
                 let next_str = &self.reverse_dict[&next_id][0];
                 // If suffix is ır, an, ılan... keep 'alın'
                 // 20085 = 'ır', 20012 = 'an' (or other variants)
                 // Check encoded strings for robustness
                 if next_str.starts_with("ır") || next_str.starts_with("an") || next_str == "nan" {
                      return tokens[0].clone();
                 }
                 // If standard possessive 'ı', let it drop
             }
        }
        
        // de (19531) / ye (19968) / başla (2206) narrowing logic
        if token_id == 19531 || token_id == 19968 || token_id == 2206 {
             let mut should_narrow = false;
             
             if i < ids.len() - 1 {
                 let next_token = &self.reverse_dict[&ids[i + 1]][0];
                 // Check for "yor" string match (covers 32621, 20041 etc)
                 if next_token.contains("yor") {
                     should_narrow = true;
                 } else if let Some(suff_forms) = self.reverse_dict.get(&ids[i+1]) {
                     if suff_forms.iter().any(|s| s.starts_with(|c| ALL_VOWELS.contains(c))) {
                          // Only for de/ye, not başla (start vowel usually narrows de/ye->di/yi but başla->başlı?)
                          // Actually 2206 (başla/başlı) only narrows for YOR usually. 
                          // "Başla" + "acak" -> "Başlayacak" (no narrowing)
                          // "Başla" + "yıp" -> "Başlayıp"
                          // So for 2206, ONLY narrow if "yor"
                          if token_id != 2206 {
                              should_narrow = true;
                          }
                     }
                 }
             }
             
             if should_narrow {
                 // For 2206: başla -> başlı (variant 1)
                 if token_id == 2206 {
                      return tokens[1].clone();
                 }
                 
                 let original = &tokens[0];
                 if original.ends_with('e') {
                     let mut s = original.clone();
                     s.pop();
                     s.push('i');
                     return s;
                 } else if original.ends_with('E') {
                     let mut s = original.clone();
                     s.pop();
                     s.push('İ');
                     return s;
                 }
             }
             return tokens[0].clone();
        }
        
        if token_id >= 100 && token_id < 2080 {
            if self.starts_with_vowel(next_token) {
                 tokens[1].clone()
            } else if token_id <= 110 && ids[i + 1] == 20034 { // ı token
                 tokens[2].clone()
            } else {
                 tokens[0].clone()
            }
        } else if token_id >= 2080 && token_id < 2315 {
             if ids[i + 1] == 20041 { // yor
                 tokens[1].clone()
             } else {
                 tokens[0].clone()
             }
        } else {
             tokens[0].clone()
        }
    }

    // Capitalize token with proper Turkish I handling
    fn capitalize_token(token: &str) -> String {
        if token.starts_with(' ') {
             // Preserve leading space
             let mut chars = token.chars();
             let first = chars.next().unwrap(); // ' '
             
             // Find first non-space
             let rest = chars.as_str();
             if rest.is_empty() { return token.to_string(); }
             
             let mut rest_chars = rest.chars();
             if let Some(c) = rest_chars.next() {
                 let cap = match c {
                     'i' => "İ".to_string(),
                     'ı' => "I".to_string(),
                     _ => c.to_uppercase().to_string(),
                 };
                 format!(" {}{}", cap, rest_chars.as_str())
             } else {
                 token.to_string()
             }
        } else {
             let mut chars = token.chars();
             if let Some(c) = chars.next() {
                 let cap = match c {
                     'i' => "İ".to_string(),
                     'ı' => "I".to_string(),
                     _ => c.to_uppercase().to_string(),
                 };
                 format!("{}{}", cap, chars.as_str())
             } else {
                 String::new()
             }
        }
    }

    pub fn decode(&self, ids: Vec<i32>) -> String {
        if ids.is_empty() { return String::new(); }
        
        let mut text_parts: Vec<String> = Vec::with_capacity(ids.len());
        let mut i = 0;
        
        while i < ids.len() {
            let token_id = ids[i];
            
            if token_id == 0 && i < ids.len() - 1 { // uppercase
                let next_token = &self.reverse_dict[&ids[i + 1]][0];
                text_parts.push(Self::capitalize_token(next_token));
                i += 2;
                continue;
            } else if token_id == 1 { // unknown
                text_parts.push("▁u▁".to_string());
            } else if let Some(tokens) = self.reverse_dict.get(&token_id) {
                if tokens.len() > 1 {
                    if token_id >= 20000 && token_id <= 20071 { // suffix
                         // Context construction (looking back up to 3 tokens)
                         let mut prev_token_str = String::new();
                         let mut j = (text_parts.len() as isize) - 1;
                         let mut tokens_checked = 0;

                         // Look back for last alphabetic part
                         let mut k = j;
                         while k >= 0 && tokens_checked < 3 {
                             let idx = k as usize;
                             if text_parts[idx].chars().any(|c| c.is_alphabetic()) {
                                 prev_token_str = text_parts[idx].clone();
                                 break;
                             }
                             k -= 1;
                         }

                         // Look back for vowel context
                         let mut vowel_context_str = prev_token_str.clone();
                         k = j;
                         tokens_checked = 0;
                         
                         // Accumulate context until we have a vowel
                         let mut found_vowel = Self::has_vowel(&vowel_context_str);
                         
                         // If the immediate prev token has no vowel, go deeper
                         if !found_vowel {
                             let mut depth = 0;
                             let mut temp_ctx = String::new();
                             
                             let mut m = (text_parts.len() as isize) - 1;
                             while m >= 0 && depth < 3 {
                                 let idx = m as usize;
                                 let part = &text_parts[idx];
                                 temp_ctx = part.clone() + &temp_ctx;
                                 if Self::has_vowel(&temp_ctx) {
                                     vowel_context_str = temp_ctx;
                                     break;
                                 }
                                 m -= 1;
                                 depth += 1;
                             }
                         }
                         
                         text_parts.push(self.select_correct_suffix(i, &ids, &vowel_context_str));
                    } else if token_id < 20000 { // root
                         text_parts.push(self.select_correct_root(i, &ids));
                    } else { // BPE (> 20071) -> Static
                         text_parts.push(tokens[0].clone());
                    }
                } else {
                    text_parts.push(tokens[0].clone());
                }
            } else {
                 text_parts.push("▁".to_string());
            }
            i += 1;
        }
        
        text_parts.join("")
    }
}
