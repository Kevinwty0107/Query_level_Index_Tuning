from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
from datetime import datetime
from bson import ObjectId, DBRef, regex
from uuid import UUID
from bson.decimal128 import Decimal128
import re


class LogEncoder(json.JSONEncoder):
    """
    Encodes MongoDB objects to string.
    """

    def __init__(self):
        super(LogEncoder, self).__init__()
        self.regex_type = type(re.compile(""))

    def default(self, o):
        if isinstance(o, ObjectId):
            return "ObjectId(%sObjectId)" % str(o)
        if isinstance(o, UUID):
            return "UUID(%sUUID)" % str(o)
        if isinstance(o, DBRef):
            return "DBRef(Field(%sField), ObjectId(%sObjectId)DBRef)" % (o.collection, str(o.id))
        if isinstance(o, datetime):
            try:
                return "ISODate(" + o.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "ZISODate)"
            except ValueError:
                return "ISODate(" + o.isoformat()[:-3] + "ZISODate)"
        if isinstance(o, (self.regex_type, regex.Regex)):
            return {"$regex": o.pattern}
        if Decimal128 and isinstance(o, Decimal128):
            return "NumberDecimal(" + str(o) + "NumberDecimal)"
        return json.JSONEncoder.default(self, o)

    def encode(self, o):
        result = super(LogEncoder, self).encode(o)
        result = result.replace('Field(', '"')
        result = result.replace("Field)", '"')
        result = result.replace('ObjectId(', 'ObjectId("')
        result = result.replace('"ObjectId(', 'ObjectId(')
        result = result.replace('ObjectId)"', '")')
        result = result.replace('ObjectId)', '")')
        result = result.replace('"DBRef(', 'DBRef(')
        result = result.replace("DBRef)\"", ')')
        result = result.replace("\"ISODate(", "ISODate(\"")
        result = result.replace("ISODate)\"", "\")")
        result = result.replace("\"UUID(", "UUID(\"")
        result = result.replace("UUID)\"", "\")")
        result = result.replace("\"NumberDecimal(", "NumberDecimal(\"")
        result = result.replace("NumberDecimal)\"", "\")")

        return result

