"""Five flat fields. This module has no access to reference answers."""
import json
import re
import unicodedata

VERSION = 'flat_audio4.v6.1'
ENUMS = {
    'gender': ['male', 'female', 'unknown'],
    'pitch_level': ['low', 'medium', 'high', 'unknown'],
    'emotion': ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted', 'other', 'unknown'],
    'emotion_intensity': ['none', 'low', 'medium', 'high', 'unknown'],
}
ATTRS = tuple(ENUMS)
FIELDS = ('transcript', *ATTRS)
SCHEMA = {'type': 'object', 'required': list(FIELDS), 'additionalProperties': False,
          'properties': {'transcript': {'type': 'string', 'minLength': 1},
                         **{k: {'enum': v} for k, v in ENUMS.items()}}}


def parse_object(raw):
    text=raw.strip()
    if text.startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',text,re.I)
        if not match:raise ValueError('incomplete code fence')
        text=match.group(1)
    def object_pairs(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('duplicate key')
            out[k]=v
        return out
    def invalid(value):raise ValueError('nonfinite JSON')
    value=json.loads(text,object_pairs_hook=object_pairs,parse_constant=invalid)
    if not isinstance(value,dict) or 'Target_JSON_Schema' in value:
        raise ValueError('V6 requires one flat JSON object')
    return value


def known(v):
    return isinstance(v, str) and bool(v.strip()) and v.strip().casefold() not in ('unknown', 'null', 'unspecified')


def normalize(field, value):
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize('NFKC', value).strip()
    if field == 'transcript':
        return value if known(value) else None
    return value.casefold() if value.casefold() in ENUMS[field] else None


def admit(raw):
    result = {'caption': {f: 'unknown' for f in FIELDS}, 'json_parseable': False,
              'format_score': 0.0, 'valid_fields': [], 'condition_fields': [], 'field_errors': {},
              'semantic_input_valid': False, 'raw_schema_valid': False, 'admission_version': VERSION}
    try:
        # Do not accept the legacy nested wrapper as a valid flat caption.
        value = parse_object(raw)
    except (ValueError, TypeError, AttributeError) as exc:
        result['parse_errors'] = [type(exc).__name__]
        return result
    result['json_parseable'] = True
    for f in FIELDS:
        v = normalize(f, value.get(f))
        if v is None:
            result['field_errors'][f] = 'missing' if f not in value else 'invalid'
        else:
            result['caption'][f] = v
            result['valid_fields'].append(f)
            if known(v): result['condition_fields'].append(f)
    result['format_score'] = 0.5 + 0.1 * len(result['valid_fields'])
    from jsonschema import Draft202012Validator
    result['raw_schema_valid'] = Draft202012Validator(SCHEMA).is_valid(value)
    cap = result['caption']
    if cap['emotion'] == 'neutral' and cap['emotion_intensity'] in ('medium', 'high'):
        cap['emotion_intensity'] = 'unknown'
        result['condition_fields'].remove('emotion_intensity')
        result['field_errors']['emotion_intensity'] = 'conflicts_with_neutral'
    result['semantic_input_valid'] = known(cap['transcript'])
    result['parse_errors'] = []
    return result


def prompt():
    return ('Listen to the audio and transcribe its spoken words faithfully. Describe the main speaker. '
            'Return ONLY a flat JSON object with the following five fields. '
            'Do not add language, explanations or other attributes. Use unknown for uncertain attributes. '
            'Neutral emotion is compatible with none/low/unknown intensity. JSON schema: '
            + json.dumps(SCHEMA, ensure_ascii=False))


def render(caption):
    cap = admit(json.dumps(caption, ensure_ascii=False))['caption']
    if not known(cap['transcript']): raise ValueError('No generated transcript for TTS')
    names = {'gender': 'Speaker gender', 'pitch_level': 'Pitch level',
             'emotion': 'Emotion', 'emotion_intensity': 'Emotion intensity'}
    parts = [f'{names[k]}: {cap[k]}' for k in ATTRS if known(cap[k])]
    return {'text': cap['transcript'], 'instruct': '; '.join(parts) or 'Read the provided text aloud.', 'language': 'Auto'}


def parse_labels(raw, fields):
    value = parse_object(raw)
    attrs, errors = {}, {}
    for f in fields:
        v = normalize(f, value.get(f))
        attrs[f] = v if v is not None else 'unknown'
        if v is None: errors[f] = 'missing_or_invalid'
    return attrs, errors
