/**
 * Copyright 2016, Emory University
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.emory.mathcs.nlp.mining.character.structure;

import java.io.Serializable;
import java.util.List;

/**
 * @author Henry(Yu-Hsin) Chen ({@code yu-hsin.chen@emory.edu})
 * @version 1.0
 * @since Mar 8, 2016
 */
public class Utterance implements Serializable {
    private static final long serialVersionUID = -7666651294661709214L;

    /* Class fields ===================================================================== */
    public int utterance_id;
    public String speaker, utterance_raw, statment_raw;
    public List<String> utterance_sentences, statement_sentences;
    public List<List<String>> 
    	tokenized_utterance_sentences, tokenized_statement_sentences,
    	utterance_sentence_annotations, statement_sentence_annotations,
    	utterance_sentence_pos_tags, statement_sentence_pos_tags,
    	utterance_sentence_dep_labels, statement_sentence_dep_labels,
    	utterance_sentence_dep_heads, statement_sentence_dep_heads,
    	utterance_sentence_ner_tags, statement_sentence_ner_tags;
    public List<StatementNode[]> statement_trees;
}
