from transformers import DonutProcessor


class DocProcessor(DonutProcessor):
    @classmethod
    def from_pretrained_with_model(cls, donut_model, *args, **kwargs):
        proc = cls.from_pretrained(*args, **kwargs)
        proc._model = donut_model
        proc._added_tokens = []
        return proc

    def json2token(self, obj, update_tokens_for_json_key=True, sort_json_key=True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if not hasattr(self, "_model"):
            raise AttributeError("Define class with model to access this method.")
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_tokens_for_json_key:
                        self.add_tokens([rf"<s_{k}>", rf"</s_{k}>"])
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(
                            obj[k], update_tokens_for_json_key, sort_json_key
                        )
                        + rf"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [
                    self.json2token(item, update_tokens_for_json_key, sort_json_key)
                    for item in obj
                ]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self._added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens):
        newly_added_num = self.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self._model.decoder.resize_token_embeddings(len(self.tokenizer))
            self._added_tokens.extend(list_of_tokens)

    def add_special_tokens(self, list_of_tokens):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(list_of_tokens))}
        )
        if newly_added_num > 0:
            self._model.decoder.resize_token_embeddings(len(self.tokenizer))
            self._added_tokens.extend(list_of_tokens)
