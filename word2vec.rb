include Math

$vocabulary_size = 0            # 词典个数
$vocabulary_max_size = 5000     # 词典最大长度
$vocabulary_hash_size = 30000
$exp_table_size = 1000
$layer1_size = 100              # 词向量长度
$alpha = 0.025                  # 学习效率

Vocabulary = Struct.new(:count, :address, :point, :word, :code, :codelength)

def train_vocabulary_list
  $vocabulary = Array.new($vocabulary_max_size)
  $vocabulary_hash = Hash.new
  $exp_table = Array.new($exp_table_size + 1)

  # 预先计算sigmoid function表
  0.upto($exp_table_size){  |i|
    # exp((i -500)/ 500 * 6) 即 e^-6 ~ e^6
    $exp_table[i] = exp((Float(i) / $exp_table_size * 2.0 - 1.0) * 6.0)
    $exp_table[i] = $exp_table[i] / ($exp_table[i] + 1.0)
  }

  train_model
end

def train_model
  puts 'Training Vocabulary List...'
  learn_vocab_from_train_file
  save_vocabulary
  init_net

  training_model

  #写入文件
  file = File.new('word_vectors.txt', 'wb')
  0.upto($vocabulary_size - 1){ |i|
    file.write($vocabulary[i][:word] + ',')
    0.upto($layer1_size){ |j| file.write( String($syn0[i * $layer1_size + j]) + ',') }
  }
  file.close
end

# 从分词文件统计词频
def learn_vocab_from_train_file
  train_file = File.open("word_vectors_tempfile.txt",'rb')
  $vocabulary_size = 0

  # 把文件中单词提取到array中
  file_content = train_file.read.split(/\W+/)
  for word in file_content
    index = index_in_vocabulary(word)

    if index == -1
      # 新词添加进单词表并计数
      new_index = add_word_to_vocabulary(word)
      $vocabulary[new_index][:count] = 1
    else
      # 已出现的词，更新词频
      $vocabulary[index][:count] += 1
    end
  end

  # :TODO 词汇量大的时候应该做一个词汇表缩减，这里先不做

  $vocabulary.compact!
  # 根据词频排序词汇表
  sort_vocabulary_by_count

  train_file.close
end

def add_word_to_vocabulary(word)
  String(word).length
  vocab = Vocabulary.new
  vocab[:word] = word
  vocab[:count] = 0
  $vocabulary[$vocabulary_size] = vocab
  $vocabulary_size += 1

  #词典扩容
  if $vocabulary_size + 1 >= $vocabulary_max_size
    $vocabulary_max_size += 100
    $vocabulary += Array.new(100)
  end

  hash = word_hash(word)
  #解决hash冲突
  hash = (hash + 1) % $vocabulary_hash_size while $vocabulary_hash[hash] != nil
  #用hash值就能找到词
  $vocabulary_hash[hash] = $vocabulary_size - 1
  $vocabulary_size - 1
end

def word_hash(word)
  hash = 0
  word = String(word)
  0.upto(word.length - 1){  |i|
    hash = hash * 257 + word[i].ord
  }
  hash % $vocabulary_hash_size
end

def index_in_vocabulary(word)
  hash = word_hash(word)
  while true
    index = $vocabulary_hash[hash]
    return -1 if index == nil
    return index if word.eql?($vocabulary[index][:word])
    hash = (hash + 1) % $vocabulary_hash_size
  end
  -1
end

def sort_vocabulary_by_count
  $vocabulary.sort!{ |x, y| y[:count] <=> x[:count] }
  # 重新计算hash
  $vocabulary_hash.clear
  0.upto($vocabulary_size - 1){ |i|
    # :TODO 删除低频词
    hash = word_hash($vocabulary[i][:word])
    hash = (hash + 1) % $vocabulary_hash_size while $vocabulary_hash[hash] != nil
    $vocabulary_hash[hash] = i
  }
end

def save_vocabulary
  file = File.new('vocabularyList.txt', 'wb')
  0.upto($vocabulary_size - 1){ |i|
    file.write("#{$vocabulary[i][:word]} #{$vocabulary[i][:count]}\n")
  }
  file.close
end

def init_net
  $syn0 = Array.new($vocabulary_size * $layer1_size)
  0.upto($layer1_size){ |i|
    0.upto($vocabulary_size){ |j|
      $syn0[j * $layer1_size + i] = (Random.new.rand - 0.5) / $layer1_size
    }
  }
  # softmax
  $syn1 = Array.new($vocabulary_size * $layer1_size)
  0.upto($layer1_size){ |i|
    0.upto($vocabulary_size){ |j|
      $syn1[j * $layer1_size + i] = 0
    }
  }
  create_huffman_tree
end

# 用词频创建huffman树
def create_huffman_tree
  # 构建大一倍的count表用来动态记录huffman树构建过程（后半段存储构建时产生的中间节点）
  creating_huffman = Array.new($vocabulary_size * 2 + 1)
  0.upto($vocabulary_size - 1){ |i| creating_huffman[i] = $vocabulary[i][:count] }
  $vocabulary_size.upto($vocabulary_size * 2 - 1){ |i| creating_huffman[i] = 0xFFFFFFF }

  # 利用数组形式来存储huffman树
  parent_node = Array.new($vocabulary_size * 2 + 1)
  huffman_tree = Array.new($vocabulary_size * 2 + 1, 0)

  temp1 = $vocabulary_size - 1
  temp2 = $vocabulary_size

  0.upto($vocabulary_size - 2){ |i|
    # 分别找到第一和第二小的权值节点
    # 第一个权值最小的节点
    if temp1 >= 0
      if creating_huffman[temp1] < creating_huffman[temp2]
        smallest1 = temp1
        temp1 -= 1
      else
        smallest1 = temp2
        temp2 += 1
      end
    else
      smallest1 = temp2
      temp2 += 1
    end

    # 第二个权值最小的节点
    if temp1 >= 0
      if creating_huffman[temp1] < creating_huffman[temp2]
        smallest2 = temp1
        temp1 -= 1
      else
        smallest2 = temp2
        temp2 += 1
      end
    else
      smallest2 = temp2
      temp2 += 1
    end

    # 合并一个中间节点
    creating_huffman[$vocabulary_size + i] = creating_huffman[smallest1] + creating_huffman[smallest2]
    parent_node[smallest1] = $vocabulary_size + i
    parent_node[smallest2] = $vocabulary_size + i
    # 以较大权值节点编码为1
    huffman_tree[smallest2] = 1
  }

  code = Array.new
  point = Array.new
  # 为单词构造huffman编码
  0.upto($vocabulary_size - 1){ |i|
    temp = i
    j = 0
    while true
      code[j] = huffman_tree[temp]
      point[j] = temp
      j += 1
      temp = parent_node[temp]
      break if temp == $vocabulary_size * 2 - 2
    end
    $vocabulary[i][:codelength] = j
    huffman_point = Array.new
    huffman_point[0] = $vocabulary_size - 2
    0.upto(j - 1){  |a|
      huffman_point[j - a] = point[a] - $vocabulary_size
    }
    $vocabulary[i][:point] = huffman_point
    $vocabulary[i][:code] = String(code.reverse.join)
  }
end

def training_model
  new_sentence = true
  neu1 = Array.new($layer1_size)
  neu1e = Array.new($layer1_size)
  file = File.open("word_vectors_tempfile.txt",'rb')
  content = file.read.split(/\W+/)

  while true
    # 从文件内容中，挑选特定长度的词组作为一个句子
    break if content.empty?

    if new_sentence
      # 选200个词作为一个句子
      sentence = content.first(200)
      0.upto(sentence.length - 1){  |i| sentence[i] = index_in_vocabulary(sentence[i]) }
      content = content.drop(200)
      position_in_sentence = 0
      new_sentence = false
    end

    current_word = sentence[position_in_sentence]
    next if current_word == -1            # 如果当前词不在词典则不处理

    0.upto($layer1_size - 1){ |i| neu1[i] = 0 }
    0.upto($layer1_size - 1){ |i| neu1e[i] = 0 }
    random_num = Random.new.rand(0..5)       # 产生0-5随机数,默认窗口长度为5

    # 训练CBOW模型

    # 扫描目标单词前后n个词，并计算向量和
    random_num.upto(10 - random_num){ |n|
      index = position_in_sentence - 5 + n
      next if index < 0 or index >= 200
      last_processed_word = sentence[index]
      next if last_processed_word == -1
      # 求向量和
      0.upto($layer1_size - 1){ |i| neu1[i] += $syn0[i + last_processed_word * $layer1_size]}
    }

    # 用softmax方法,遍历huffman树
    0.upto($vocabulary[current_word][:codelength] - 1){ |i|
      function = 0

      # point应该记录的是huffman的路径。找到当前节点，并算出偏移
      l2 = $vocabulary[current_word][:point][i] * $layer1_size
      # Propagate hidden -> output
      0.upto($layer1_size - 1){ |j| function += neu1[j] * $syn1[j + l2] } # 计算内积

      next if function <= -6 or function >= 6
      # sigmoid函数, 查预先算好的exp表得到
      function = $exp_table[(function + 6) * ($exp_table_size / 6 / 2)]

      # gradient 的一部分
      g = (1 - Float($vocabulary[current_word][:code][i]) - function) * $alpha

      # 反向传播误差，从huffman树传到隐藏层。下面就是把当前内节点的误差传播给隐藏层，syn1[c + l2]是偏导数的一部分。
      0.upto($layer1_size - 1){ |j| neu1e[j] += g * $syn1[j + l2] }

      # 更新当前内节点的向量，后面的neu1[c]其实是偏导数的一部分
      0.upto($layer1_size - 1){ |j| $syn1[j + l2] += g * neu1[j] }
    }

    # 跟新周围词语的词向量
    random_num.upto(10 - random_num){ |n|
      next if n == 5          # 不更新当前词的向量，只更新周围词语
      index = position_in_sentence - 5 + n
      next if index < 0 or index >= 200
      last_processed_word = sentence[index]
      next if last_processed_word == -1
      # 更新词向量
      0.upto($layer1_size - 1){ |j| $syn0[j + last_processed_word * $layer1_size] += neu1e[j] }
    }

    # 处理下一个中心词语
    position_in_sentence += 1
    # 如果当前语句处理完则处理下一组词语
    if position_in_sentence >= 200
      new_sentence = true
      next
    end
  end
  file.close
end
