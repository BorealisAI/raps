# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
while true; do
    clear
    seq $1 $2 | xargs -L1 ts -o | xargs tail -n1 \
        | awk 'NR % 2 == 0'
    sleep 5
done
