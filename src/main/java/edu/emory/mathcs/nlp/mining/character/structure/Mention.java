/**
 * Copyright 2016, Emory University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.emory.mathcs.nlp.mining.character.structure;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author 	Henry(Yu-Hsin) Chen ({@code yu-hsin.chen@emory.edu})
 * @version	1.0
 * @since 	Nov 12, 2016
 */
public class Mention implements Comparable<Mention>, Serializable{
	private static final long serialVersionUID = 6455404540781499704L;
	public static final String ROOT_TAG = "@#r$%";

	public static Mention root() {
		return new Mention(0, 0, null, ROOT_TAG, ROOT_TAG);
	}
	
	/* Source Information */
	private int head_token_id; // Statement token	
	private Utterance utterance;
	
	/* Mention Information */
	private int mention_id;
	private Mention antecedent;
	private List<String> tokens, annotations;
	
	public Mention(int mention_id, int head_token_id, Utterance utterance) {
		this.mention_id = mention_id;
		this.utterance = utterance;
		this.head_token_id = head_token_id;
		
		this.tokens = new ArrayList<>();
		this.annotations = new ArrayList<>();
	}
	
	public Mention(int mention_id, int head_token_id, Utterance utterance, String token, String annotation) {
		this(mention_id, head_token_id, utterance); addToken(token, annotation); 
	}
	
	public boolean isRoot() {
		return mention_id == 0;
	}
	
	public boolean isAntecedentOf(Mention mention) {
		return this == mention.getAntecedent();
	}
	
	public int size() {
		return tokens.size();
	}
	
	public int getID() {
		return mention_id;
	}
	
	public int getHeadTokenID() {
		return head_token_id;
	}
	
	public Utterance getUtterance() {
		return utterance;
	}
	
	public Mention getAntecedent() {
		return antecedent;
	}
	
	public List<String> getTokens() {
		return tokens;
	}
	
	public List<String> getAnnotations() {
		return annotations;
	}
	
	public void setHeadTokenID(int head_token_id) {
		this.head_token_id = head_token_id;
	}
	
	public void setUtterance(Utterance utterance) {
		this.utterance = utterance;
	}
	
	public void setAntecedent(Mention antecedent) {
		this.antecedent = antecedent;
	}
	
	public void addToken(String token, String annotation) {
		tokens.add(token); annotations.add(annotation);
	}

	@Override
	public int compareTo(Mention o) {
		return this.mention_id - o.getID();
	}
}
