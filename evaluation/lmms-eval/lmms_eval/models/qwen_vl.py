import uuid
import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None


@register_model("qwen_vl")
class Qwen_VL(lmms):
    """
    Qwen_VL Model
    https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/evaluate_vqa.py
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen-VL",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self._device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device if device != "cuda" else "cuda:0")
            self._device_map = str(self._device)

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.max_new_tokens = max_new_tokens
        self._processor = None

        self._config = AutoConfig.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._multimodal_model_type = self._config.model_type
        self._multimodal_model_cls = {
            "qwen2_vl": Qwen2VLForConditionalGeneration,
            "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
            "qwen3_vl": Qwen3VLForConditionalGeneration,
        }.get(self._multimodal_model_type)
        self._is_modern_qwenvl = self._multimodal_model_cls is not None

        if self._is_modern_qwenvl:
            dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
            if self._multimodal_model_cls is None:
                raise ImportError(
                    f"Installed transformers does not provide a conditional generation class for {self._multimodal_model_type}"
                )
            self._processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
            self._tokenizer = self._processor.tokenizer
            self._model = self._multimodal_model_cls.from_pretrained(
                pretrained,
                trust_remote_code=trust_remote_code,
                torch_dtype=dtype,
                device_map=self._device_map,
            ).eval()
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self._model = AutoModelForCausalLM.from_pretrained(pretrained, device_map=self._device_map, trust_remote_code=trust_remote_code).eval()
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token_id = self.tokenizer.eod_id
            self.prompt = "<img>{}</img>{}"

        self._config = self.model.config
        self.model.tie_weights()

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return getattr(self.tokenizer, "eod_id", self.tokenizer.eos_token_id)

    @property
    def max_length(self):
        return getattr(self._config, "max_position_embeddings", None)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input_data):
        new_list = []
        for item in input_data:
            for nested in item:
                new_list.append(nested)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        if self._is_modern_qwenvl:
            raise NotImplementedError(
                f"Loglikelihood is not implemented for {self._multimodal_model_type} in this wrapper"
            )

        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            query = []
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                query.append({"image": f"/tmp/{name}.png"})

            context_query = [_ for _ in query]
            context_query.append({"text": contexts})
            query.append({"text": contexts + continuation})

            context_query = self.tokenizer.from_list_format(context_query)
            query = self.tokenizer.from_list_format(query)

            _, context_tokens = make_context(
                self.tokenizer,
                context_query,
                history=None,
                system="You are a helpful assistant",
                max_window_size=self.model.generation_config.max_window_size,
                chat_format=self.model.generation_config.chat_format,
            )
            context_tokens = torch.tensor([context_tokens])

            _, continuation_tokens = make_context(
                self.tokenizer,
                query,
                history=None,
                system="You are a helpful assistant",
                max_window_size=self.model.generation_config.max_window_size,
                chat_format=self.model.generation_config.chat_format,
            )
            continuation_tokens = torch.tensor([continuation_tokens]).to(self.model.device)
            attn_mask = torch.ones_like(continuation_tokens).to(self.model.device)
            labels = continuation_tokens.clone().to(self.model.device)
            labels[:, : context_tokens.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=continuation_tokens, labels=labels, attention_mask=attn_mask)
            loss = outputs.loss
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = continuation_tokens[:, context_tokens.shape[1] :]
            greedy_tokens = greedy_tokens[:, context_tokens.shape[1] : continuation_tokens.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def _generate_multimodal_qwen(self, context, gen_kwargs, doc_to_visual, doc_id, task, split):
        visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
        visuals = self.flatten(visuals)

        content = [{"type": "image"} for _ in visuals]
        content.append({"type": "text", "text": context})
        message = [{"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = self._processor(text=[prompt], images=[visuals], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generation_kwargs = dict(gen_kwargs)
        if "max_new_tokens" not in generation_kwargs:
            generation_kwargs["max_new_tokens"] = self.max_new_tokens
        if "max_length" in generation_kwargs and "max_new_tokens" in generation_kwargs:
            generation_kwargs.pop("max_length")
        if "temperature" not in generation_kwargs:
            generation_kwargs["temperature"] = 0
        if "top_p" not in generation_kwargs:
            generation_kwargs["top_p"] = None
        if "num_beams" not in generation_kwargs:
            generation_kwargs["num_beams"] = 1
        generation_kwargs.pop("until", None)
        generation_kwargs.pop("image_aspect_ratio", None)
        if "do_sample" not in generation_kwargs:
            generation_kwargs["do_sample"] = False

        output_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)

            if self._is_modern_qwenvl:
                assert len(chunk) == 1, f"{self._multimodal_model_type} wrapper currently supports batch_size=1"
                res.append(
                    self._generate_multimodal_qwen(
                        contexts[0], all_gen_kwargs[0], doc_to_visual[0], doc_id[0], task[0], split[0]
                    )
                )
                pbar.update(1)
                continue

            task_name = task[0]
            split_name = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task_name][split_name][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            visual_paths = []
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")

            gen_kwargs = dict(all_gen_kwargs[0])
            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected gen_kwargs['until'] to be Union[str, list], got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            query = []
            if len(visual_paths) == 0:
                for context in contexts:
                    query.append({"text": context})
            else:
                for visual_path, context in zip(visual_paths, contexts):
                    query.append({"image": visual_path})
                    query.append({"text": context})

            questions = self.tokenizer.from_list_format(query)
            input_ids = self.tokenizer(questions, return_tensors="pt", padding="longest")
            if "image_sizes" not in gen_kwargs:
                try:
                    gen_kwargs["image_sizes"] = [visuals[0].size]
                except Exception:
                    gen_kwargs["image_sizes"] = None
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            input_ids = {k: v.to(self.model.device) for k, v in input_ids.items()}
            output_ids = self.model.generate(**input_ids, **gen_kwargs)
            responses = self.tokenizer.batch_decode(output_ids[:, input_ids["input_ids"].shape[1] :], skip_special_tokens=True)
            for response in responses:
                for term in until:
                    if term:
                        response = response.split(term)[0]
                res.append(response)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
