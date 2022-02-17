MODEL_PATH=$1

# Many to One
# DATA_DIR=../data/iwslt14/
# for SRC in ar  de  es  fa  he  it  nl  pl  pt  ro  ru  sl  tr  zh; do
#     TGT=en
#     FSRC=${DATA_DIR}/tok/test.${SRC}-${TGT}.${SRC}
#     FTGT=${DATA_DIR}/preprocessed/${SRC}/test.${TGT}
#     FOUT=${MODEL_PATH}/results/test.${SRC}-${TGT}.${TGT}
#     mkdir -p ${MODEL_PATH}/results

#     cat $FSRC | python scripts/truncate.py | \
#     fairseq-interactive ${DATA_DIR}/data-bin \
#         --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH/checkpoint_best.pt \
#         --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
#         --lang-pairs ar-en,es-en,he-en,nl-en,pt-en,ru-en,tr-en,de-en,fa-en,it-en,pl-en,ro-en,sl-en,zh-en \
#         --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
#         --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
#     grep -P "^H" | cut -f 3- > $FOUT

#     cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu

# done

# python scripts/get_mean_bleu.py --path ../checkpoints/iwslt_base/many-to-one/results/ --data iwslt

# One to Many 
# DATA_DIR=../data/iwslt14/
# for TGT in ar  de  es  fa  he  it  nl  pl  pt  ro  ru  sl  tr  zh; do
#     SRC=en
#     FSRC=${DATA_DIR}/tok/test.${TGT}-${SRC}.${SRC}
#     FTGT=${DATA_DIR}/preprocessed/${TGT}/test.${TGT}
#     FOUT=${MODEL_PATH}/results/test.${SRC}-${TGT}.${TGT}
#     mkdir -p ${MODEL_PATH}/results

#     cat $FSRC | python scripts/truncate.py | \
#     fairseq-interactive ${DATA_DIR}/data-bin \
#         --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH/checkpoint_best.pt \
#         --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
#         --lang-pairs en-ar,en-es,en-he,en-nl,en-pt,en-ru,en-tr,en-de,en-fa,en-it,en-pl,en-ro,en-sl,en-zh \
#         --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
#         --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
#     grep -P "^H" | cut -f 3- > $FOUT

#     cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu

# done

# python scripts/get_mean_bleu.py --path ../checkpoints/iwslt_base/one-to-many/results/ --data iwslt --nen



## Latin to One
# DATA_DIR=../data/iwslt14/
# for SRC in de es it nl pl pt ro sl tr; do
#     TGT=en
#     FSRC=${DATA_DIR}/tok/test.${SRC}-${TGT}.${SRC}
#     FTGT=${DATA_DIR}/preprocessed/${SRC}/test.${TGT}
#     FOUT=${MODEL_PATH}/results/test.${SRC}-${TGT}.${TGT}
#     mkdir -p ${MODEL_PATH}/results

#     cat $FSRC | python scripts/truncate.py | \
#     CUDA_VISIBLE_DEVICES=3 fairseq-interactive ${DATA_DIR}/data-bin \
#         --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH/checkpoint_best.pt \
#         --langs de,es,it,nl,pl,pt,ro,sl,tr,en \
#         --lang-pairs de-en,es-en,it-en,nl-en,pl-en,pt-en,ro-en,sl-en,tr-en \
#         --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
#         --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
#     grep -P "^H" | cut -f 3- > $FOUT

#     cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu

# done

# De -> One
DATA_DIR=../data/iwslt14/
for SRC in de ; do
    TGT=en
    FSRC=${DATA_DIR}/tok/test.${SRC}-${TGT}.${SRC}
    FTGT=${DATA_DIR}/preprocessed/${SRC}/test.${TGT}
    FOUT=${MODEL_PATH}/results/test.${SRC}-${TGT}.${TGT}
    mkdir -p ${MODEL_PATH}/results

    cat $FSRC | python scripts/truncate.py | \
    CUDA_VISIBLE_DEVICES=3 fairseq-interactive ${DATA_DIR}/data-bin \
        --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH/checkpoint_best.pt \
        --langs de,en \
        --lang-pairs de-en \
        --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
        --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
    grep -P "^H" | cut -f 3- > $FOUT

    cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu

done