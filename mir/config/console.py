# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from logging import DEBUG, INFO, Logger

nfo_obj = Logger(INFO)
dbuq_obj = Logger(DEBUG)

nfo = nfo_obj.info
dbuq = dbuq_obj.debug
