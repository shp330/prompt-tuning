# Prompt Tuning Implementation Details

This document details how the code works, where it lives, and some of the goals
of the implementation.

*   [Modeling in T5X](#modeling-in-t5x)
*   [Creating Prompts](#creating-prompts)
*   [Adding Prompts to Models](#adding-prompts-to-models)
*   [Prompt-Only Updates](#prompt-only-updates)
*   [Gin, Factories, and Flaxformer](#gin-factories-and-flaxformer)
*   [Partial Loading](#partial-loading)
*   [Partial Saving](#partial-saving)
*   [Testing](#testing)

## Vocabulary and Notation

*   **B ::** The batch size.
*   **T ::** 输入的序列维度，即单个样本中的 token 数量.
*   **H ::** The hidden/embedding dimension of the model.
*   **P ::** The length of the prompt.
*   **Model Tuning ::** Transfer a model to a new task by updating all the
    parameters in the models. Commonly called fine-tuning.
*   **前缀语言模型（Prefix-LM）**：一种期望在批次字典中同时包含输入特征和目标特征的语言模型。
    这类模型支持两种注意力掩码：因果注意力掩码和前缀注意力掩码.
*   **因果注意力掩码（Causal Attention Mask）**：一种注意力掩码机制，其中每个时间步（timestep）仅能关注到它之前的时间步。See mask #2 in Figure 3 of the
    [T5 paper](https://arxiv.org/pdf/1910.10683.pdf).
*   **前缀注意力掩码**：一种注意力掩码机制，其中输入的某段前缀部分支持双向注意力可见性
    （即前缀内部的时间步可以互相关注） See mask #3 in
    Figure 3 of the [T5 paper](https://arxiv.org/pdf/1910.10683.pdf). *Note:*
    The use of this mask implies the use a `Prefix-LM` but the use of a
    `Prefix-LM` doesn't imply the use of this mask.
*   **表述器（verbalizers）**：用于表示类别标签的字符串
    [(Schick and Schütze, 2021)](https://arxiv.org/pdf/2001.07676.pdf).

## Modeling in T5X

当涉及 T5X 框架的模型构建时，存在三个层级的建模结构，

1.  [T5X/models.py](https://github.com/google-research/t5x/tree/main/t5x/models.py) ::
    最外层是 T5X 框架定义的模型类。
    这些是标准的 Python 类（并非 Flax 框架 `nn.Module` 的子类），包含 `predict_batch`（批量预测）和 
    `compute_logits`（计算对数概率）等方法。它们负责与底层的 `Flax` 模块进行交互，调用 Flax 框架的 `init`（初始化）和 
    `apply`（前向计算）方法来完成核心逻辑的执行。
2.  [Flaxformer EncoderDecoder](https://github.com/google/flaxformer/tree/main/flaxformer/architectures/t5/t5_architecture.py) ::
    这一层是 Flaxformer 库中的模块，负责管控底层各个组件的执行流程。
    例如，`EncoderDecoder` 类不仅包含 `encode`（编码）和 `decode`（解码）方法，
    持有实际编码器（encoder）和解码器（decoder）模型的引用，
    还定义了 `__call__` 方法来确保这些组件按正确的顺序被调用。
    此外，这一层还负责生成注意力掩码 —— 这也是我们需要与该层级进行交互的主要原因。
3.  [Flaxformer Encoder and Decoder](https://github.com/google/flaxformer/tree/main/flaxformer/architectures/t5/t5_architecture.py) ::
    这一层是真正执行任务的 `Flaxformer` 模块（例如 `Encoder` 类就属于这一层）。
    我们对这一层进行了定制修改，实现了将提示参数实际添加到输入中的功能。
    
## Creating Prompts 提示参数的构建

我们的提示方案采用了一个提示模块来生成提示参数，并将这些参数直接添加到经过嵌入处理的输入中；
而非采用带有可更新嵌入向量的特殊虚拟词元。

我们提示（prompt）机制的核心实现在
[prompts.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/prompts.py) 文件中.
核心的 `Prompt` 模块接收输入的 token 张量 `[B, T]` 和嵌入后的输入张量 `[B, T, H]` 并返回一个未按批次展开的提示变量 `[P, H]`.

## Using Prompt in Training

[train/prompts.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/prompts.py)
包含了一些用于在训练过程中插入提示（prompts）的辅助模块。实际的提示实现被设计为仅返回提示本身，以最大限度地提升其使用灵活性。这些训练模块负责将提示与嵌入后的输入实际结合起来。

## Adding Prompts To Models

我们所有的提示层（prompting layers）都基于 Flaxformer 构建。它们通常是小型子类，通过重写某个方法来插入对提示模块的调用。

此外，还有一些 Flaxformer 的子类，其主要职责是生成经过更新的注意力掩码（attention mask），使其能够感知我们的提示（prompt-aware）。

### Adding to Encoders

[train/layers.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/layers.py) 中的 `PromptEncoder` 类是我们定义的一个子类，用于将提示 `prompt` 添加到嵌入后的输入中。可以看出，这里唯一实质性的改动是：新增了一个类属性，用于创建提示模块（prompt module）；在前向过程中显式调用了该提示模块。

[train/layers.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/layers.py)中的 `PromptEncoderDecoder` 类则是我们用于更新注意力掩码（masking）的子类。在编码器（encoder）一侧，我们使用了新定义的
`prompt_encoder_attention_mask` 函数来生成新的编码器掩码。这使我们能够对提示部分实现更精细的掩码策略（例如限制提示 token 之间的交互等）。 而在解码器（decoder）一侧，只需将其视为输入中多了 `P` 额外的 token 即可。由于解码器始终对所有编码器输出 token 具有完全可见性（full visibility），因此无需对解码器侧的掩码进行特殊处理。

### Adding to Decoders

[train/layers.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/layers.py) 中的`PromptDecoder`类用于将提示（prompts）直接添加到解码器（decoder）本身. 
`PromptDecoderOnly`类则负责处理与解码器相关的更新后注意力掩码（updated masking）.

*注* 理论上可以将 `PromptDecoder` 与 `PromptEncoder` 结合起来，创建一个名为
`PromptEncoderPromptDecoder` 的新类，从而在模型的编码器和解码器两侧都应用提示. 
实现该类时，需要借鉴 `PromptEncoderDecoder` 类中的掩码技术来构建 `encoder_decoder_mask`
用于控制解码器对编码器输出的可见性，特别是包含提示 token 时）；同时采用
`PromptDecoderOnly` 类中的方法来生成 `decoder_mask`（用于管理解码器内部的因果掩码或提示相关掩码）.
类似地，也可以创建一个仅在解码器侧应用提示的 encoder-decoder 模型变体. 
不过，截至目前，我们尚未有实际需求去实现这些组合类.

#### Decoding with Prompts 带提示的解码器

大多数提示微调（prompt tuning）都可以在“幕后”完成, 
也就是说，用户除了进行配置之外，无需改变与模型模块的交互方式即可使用提示. 
然而，当对一个在解码器本身应用了提示的模型进行解码时，就需要做出一些调整。
这是因为提示的存在会影响多个解码相关机制，例如, 自回归缓存（autoregressive cache）的大小；
单次前向传播中需要填充的时间步数（timesteps）；
以及其他与序列生成相关的内部状态管理. 为应对这些变化，
我们在 [train/models.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/models.py) 中实现了 `PromptPrefixLanguageModel` 类，
专门用于支持在解码过程中包含提示前缀（prompt prefix）的语言模型解码. 
该类封装了上述调整逻辑，确保在生成时正确地将提示视为输入序列的固定前缀，并相应地初始化和更新缓存与掩码等组件。

## Prompt-Only Updates 仅更新提示

提示微调（prompt tuning）的核心目标是：在保持原始模型参数冻结的前提下，
仅让提示（prompt）部分可学习和更新。因此，我们需要一种机制，
在模型权重更新时，只将梯度应用于提示相关的参数。 我们的解决方案基于
[flax.optim.MultiOptimizer](https://flax.readthedocs.io/en/latest/flax.optim.html#flax.optim.MultiOptimizer).
该优化器接受一个由 （遍历器 traversal，优化器 optimizer）`元组` 组成的序列。
其中：遍历器（traversal） 接收一个参数名称（通过将嵌套结构替换为 `/` 将名称展平），
并对其应用一个过滤函数. 如果过滤函数返回 `True` 则对应的优化器将用于更新该参数. 
对于未被任何优化器覆盖的参数，其 `param_state` 将为 `None` ，即不会被更新.

在我们的实现中，遍历器的过滤函数由一组正则表达式构成。
只要参数名匹配其中任意一个正则表达式，该参数就会被选中并进行更新。

要使用这个多优化器并通过 Gin 进行配置，我们需要几个工具函数.

*   [train/utils.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/utils.py) 中的 `match_any`  函数用于创建一个过滤函数，当传入的 `path
    `参数与提供的任意一个正则表达式匹配时，该过滤函数会返回 `True`.
*   我们在 [train/optim.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/utils.py)
    中维护了一个 `MultiOptimizer` 的分支。在 T5X 中，参数分区
   （即决定在模型并行任务中，如何将参数拆分到多个计算节点上）采用自底向上的实现方式：
    Flax 模块会在输出自身参数的同时，一并输出一个 `params_axes` 集合.
    这个集合用于为参数维度分配逻辑名称，而这些逻辑名称又被用于制定参数分区规则. 
    这种配置方式要求我们必须使用 T5X 内置捆绑的 [Adafactor](https://proceedings.mlr.press/v80/shazeer18a/shazeer18a.pdf)优化器版本，
    而非 `Flax` 中的版本. 该版本 `Adafactor` 的 API 与默认版本（Flax 中的版本）存在细微差异，
    而本次对 `MultiOptimizer` 的定制修改，正是为了适配这种差异。此外，由于 Gin 无法绑定可变参数
   （即可以接受多个输入值的参数，例如 Python 中的 `*args`)，该派生版本还修改了构造函数的签名：将
     `*travsreals_and_optimizers` 转换为一个接收列表的普通参数，后续再对该列表进行解包处理，以此规避 Gin 的限制.


*注意:* 模型部分训练（局部训练）会生成一个特殊的检查点 —— 在该检查点中，
历史优化器状态会被设为 `None` ，且最终检查点中会缺失这些状态信息. 
这意味着，若想开启新一轮训练（训练另一组不同的变量，并重新加载其对应的优化器状态） 
直接使用该检查点是无法实现的，必须手动将包含优化器状态的原始检查点， 与包含已训练参数的新检查点进行合并才行.
截至 `2022/01/07` 无法通过学习率调度策略来模拟模型部分训练。原因在于，即便将学习率设置为 0，
学习率调度仍会导致优化器状态发生变更，从而无法满足部分训练的要求.

## Partial Loading 部分加载

在 Flax 和 T5X 中，通常假定 配置中定义的优化器状态结构 与 检查点中保存的优化器状态结构 完全一致. 
然而，在 提示微调（prompt tuning） 场景下，这一假设被打破：模型新增了 `prompt` 参数（如 prompt_embeddings），
而这些参数在原始检查点中并不存在。
为解决此问题，T5X 提供了 `utils.RestoreCheckpointConfig` 中的 `assignment_map` 字段，
允许我们显式声明哪些参数无需从检查点加载。`assignment_map` 是一个元组序列，
每个元组形如 (config_name_pattern, checkpoint_name)。其主要用途是关联模型与磁盘上检查点之间不同的变量名。
正常情况下，元组中的第一个元素是配置文件定义的优化器中的变量名，第二个元素是检查点定义的优化器中的变量名. 
如果我们将第二个元素设为 `None`，就表示将该参数标记为`跳过加载`，不会尝试从磁盘检查点中读取该参数。
配合 `fallback_scratch` 参数（该参数会对所有未从磁盘检查点加载的参数，使用模块的默认初始化方法进行补全初始化），
我们就能实现：从检查点中加载主模型的参数，同时从零开始初始化提示参数（prompt）。

为简化操作，我们通常会使用 `((r".*prompt.*", None),)` 这样的元组，来查找所有提示相关的变量。

*注意：* T5X 中 `assignment_map` 的相关代码使用了 `re.fullmatch` 方法进行匹配，
这意味着提供的正则表达式必须与字符串完整匹配。也就是说，`prompt` 前后的 `.*` 是必不可少的。

## Partial Saving 部分保存
在训练大型模型（例如 T5-XXL）时，尽管提示（prompt）本身可能看起来参数量不小
（例如长度为 100、嵌入维度为 4096 时，共有 409,600 个参数），但实际上它仅占模型总参数的极小一部分——
在这个例子中仅为 0.0037%。因此，每次保存完整的模型检查点（其中绝大部分参数与原始预训练模型完全相同）是一种严重的存储浪费。

我们通过 [train/utils](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/utils.py) 
中自定义的 `Checkpointer` 子类，在一定程度上缓解了这个问题。
在保存整个模型之前，该子类会将所有变量名与正则表达式列表中任意一个匹配的参数，保存为 Numpy 数组。
生成的文件会存储在 `${MODEL_DIR}/numpy_checkpoints/checkpoint_{$step}` 目录下，
文件名是扁平化后的参数作用域（将路径中的 `/` 替换为 `.`）。我们将这些 Numpy 格式的检查点保存到独立目录中，
避免受到 T5X 检查点保留策略配置的影响。

*注意：*尽管从技术上来说，我们的检查点只需要这些 Numpy 文件即可，但目前我们仍然会执行完整的 T5X 检查点保存流程。
原因在于，当 T5X 从任务抢占中断中恢复训练时，它会检查自身保存的数据格式，并加载该检查点（包含训练步数等关键信息）
以继续训练。若要跳过完整 T5X 检查点保存，我们需要做一系列修改：在主训练函数中重写相关逻辑，使其能够识别 Numpy 
目录并用于恢复训练；同时还需要重写检查点加载逻辑，先加载默认 T5X 检查点，再用从 Numpy 文件中加载的值覆盖
对应变量。这样做虽然能减少模型保存的耗时，但考虑到训练代码的部分模块缺乏可配置性，这样的改造并不值得。
因此，我们将保留的检查点数量设置为 1 —— 这样既能够支持从任务抢占中断中恢复训练，又能在不生成大型模型多个副本的前提下，
获取所有训练步数对应的模型状态（仅通过 Numpy 小文件）。

## Gin, Factories and Flaxformer

理解 Gin 工作原理的一个简单方法，是将它想象成 Python 中的 `functools.partial`。
一个支持 Gin 配置的对象（Gin configurable）对应着一个函数，当 Gin 解析配置文件时，
会将配置文件中定义的参数应用到该函数上。例如，我们有如下一个支持 Gin 配置的对象：

```python
def my_func(x, b, c):
  pass
```

和配置文件

```python
my_func:
  x = 2
  b = "hello"
  c = 34
```

我们可以这样理解：Gin 会将这份配置应用到我们的函数上，之后我们就可以不带任何参数直接调用该函数，示例如下:

```python
# gin's configuration parsing applies arguments
my_func = functools.partial(my_func, x=2, b="hello", c=34)
# Then when we call it we don't need to include arguments
my_func()
```

`Flaxformer` 在其模型配置中大量运用了这一思想。该框架要求绝大多数类属性都必须是工厂（函数/对象）—— 也就是说，
你不能直接将一个 `nn.Module` 实例传入 `Flaxformer` 模型，而是需要传入一个无参调用时能返回正确 `nn.Module` 实例的函数。
在 `gin` 配合使用的场景下，他们的实现方式是：将类本身作为工厂，通过 `gin` 为该类绑定所有所需的构造参数。
随后，当 `Flaxformer` 调用 `.setup` 方法时，会逐一调用这些工厂函数，最终返回你所指定的 `nn.Module` 实例。

我们在提示调优（Prompt Tuning）中也遵循了这一模式：所有与 `Flaxformer` 相关的配置，
绝大多数要么是已绑定构造参数的可调用对象，要么是对实际函数的闭包（该闭包会通过工厂调用返回）。

## Testing 测试说明

要运行测试用例，请先以 `[test]` 选项安装依赖包，然后在克隆仓库的根目录下执行 `pytest` 命令。

在条件允许的情况下，我们会尝试对所有基于 `JAX` 编写的代码，在 `jax.jit` 装饰（即时编译）环境下进行测试。
这有助于发现一些在其他测试场景下难以察觉的隐藏 Bug。

部分方法（例如提示参数初始化函数）在被 `jit` 编译时，需要将其部分参数标记为静态参数（static）。
这类函数通常会接收诸如 `shape` 之类的参数 —— 在 `Flax` 框架中使用时，它们能正常工作；
但直接对这类函数应用 `jit` 编译，会导致 shape 参数被转换为追踪对象（tracer object），
原本可直接访问的形状信息，会变成无法直接使用的张量值。

当你使用 Mock 模拟对象来测试某个可注入模块 / 函数是否被正确调用时
（例如，测试用例以调用 `.assert_called_once_with(...)` 断言方法结束），
不能对被测试方法进行 `jit` 编译。否则会抛出追踪对象泄露（tracer object has leaked）相关错误。
不过，你可以在 `jit` 编译后的方法中使用 Mock 来控制输出结果（这种场景下不会出现泄露问题）。

我们会尽可能使用 `mock.create_autospec` 和 `mock.patch.object(obj, "attr", autospec=True)` 创建 Mock 对象。
它们能验证 Mock 对象是否被调用了自身不存在的方法，从而提升测试的严谨性。但这两个方法也可能引发问题，
尤其是当你对一个类启用 `autospec` 且返回 `instance=True` 的实例对象时：如果你尝试调用该 Mock 实例的
`.assert_called_once_with` 方法，会因为 `self` 参数未被正确处理而抛出错误。
在这种情况下，你需要使用 `self.assertEqual` 和 `mock.call` 来校验 Mock 对象的调用情况是否等价。
 
我们提供了两个测试工具类（`ArrayEqualMatcher` 和 `ArrayAllCloseMatcher`），
用于辅助对接收 `JAX/Numpy` 数组作为参数的函数调用进行断言。
`.assert_called_once_with` 方法无法正确处理非同一对象但值相同的数组参数 —— 它在执行相等性检查时，
在调用（可能成本高得多的）`__eq__` 方法前，会先进行 `is` 检查。
这意味着：如果 Mock 被调用时传入的数组对象，与你在断言中使用的数组对象是**同一个**对象，测试会通过；
但如果它们是拥有相同值的**不同**对象，测试会失败。而这两个 `ArrayMatcher` 工具类定义了 `__eq__` 方法，
使用 `np.(array_equal|allclose)` 来进行数组对比。通过将预期调用参数（即 `.assert_called_once_with` 中的传入参数）
中的数组用该工具类包装， 我们就能在包含数组参数的断言中，使用常规的断言方法并得到正确结果。

### Longer Running, More Integration-Style Tests 耗时较长的集成式测试

文件 [train/train_test.py](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/train/train_test.py)
中包含若干耗时较长的测试用例，这类测试更偏向于大规模集成测试，
用于验证模型、网络层、配置文件以及提示参数（prompt）能否协同正常工作。其中包含两个核心测试：
一是验证当模型被更新时，仅有提示变量（prompt variable）发生变更（其余模型参数保持不变）；
二是验证模型权重（除提示权重外）能否从检查点中被正确加载。

