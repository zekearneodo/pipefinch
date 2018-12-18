# tools to read setting files from recorder (.txt)
import warnings
import logging

logger = logging.getLogger('pipefinch.recorder.core.setting')

line_special_char_section = {
    '##': 'activator',
    '%%': 'detector',
    '&&': 'effector',
}

# what is a line encoding


def what_is_line(line: str, sect_char_dict: dict=line_special_char_section) -> str:
    section_keys = [v for (k, v) in sect_char_dict.items() if k in line]

    if len(section_keys) == 1:
        line_type = section_keys[0]
    elif len(section_keys) > 1:
        raise ValueError(
            'More than one section key special char found in line {}'.format(line))

    elif line.strip() == '':
        line_type = 'blank'
    elif ':' in line.strip():
        line_type = 'key'
    elif len(section_keys) == 0:
        line_type = 'value'
    return line_type


def make_chan_info_dict(settings_dict: dict) -> dict:
    ch_info_tuple = tuple()
    for ch_id, ch_info_string in zip(settings_dict['channels'], settings_dict['channel info']):
        ch_info_list = ch_info_string.split(',')
        ch_info_dict = {'chan_id': ch_id,
                        'name': ch_info_list[1].strip(),
                        'save': ch_info_list[2].strip().split(' ')[1],
                        'gain': float(ch_info_list[3].strip().split(' ')[1]),
                        'rig': ch_info_list[0].strip()}
        ch_info_tuple += (ch_info_dict, )
    settings_dict['chan_info'] = ch_info_tuple
    return settings_dict

# read values of a line


def read_values(all_lines: list, values_tuple: tuple):
    line = all_lines.pop(0)
    line_type = what_is_line(line)
    if line_type is 'value':
        values_tuple += (line.strip(), )
        #logger.debug('line is value' + str(values_tuple))
        values_tuple = read_values(all_lines, values_tuple)
    elif line_type is 'blank':
        pass
        #logger.debug('line is blank')
        # logger.debug(values_tuple)
    else:
        raise ValueError(
            'Weird line {} found while reading parameter values'.format(line))
    return values_tuple

# Read a block describing a processor


def read_recorder_processor(all_lines: list, processor_dict: dict) -> dict:
    line = all_lines.pop(0)
    #logger.debug('reading processor line {}'.format(line))
    if '{' in line:
        try:
            key, val = line.strip().strip('\\').strip('{').split(':', 1)
            #logger.debug('read {}: {}'.format(key, val))
            processor_dict[key] = val
        except ValueError:
            warnings.warn(
                'Failed to parse key: val reading recorder processor settings line {}'.format(line))
        processor_dict = read_recorder_processor(all_lines, processor_dict)
    elif '}' in line:
        #logger.debug('end of processor reading')
        pass
    else:
        warnings.warn(
            'Unbecoming line while reading recorder processor settings line {}'.format(line))
        processor_dict = read_recorder_processor(all_lines, processor_dict)
    return processor_dict

# Read through the file beginning at top level


def process_top_level(all_lines: list, set_dict: dict):
    try:
        line = all_lines.pop(0)
        line_type = what_is_line(line)

    except IndexError as err:
        if len(all_lines) == 0:
            line_type = 'end'
        else:
            raise

    #logger.debug('{} is {}'.format(line, line_type))
    if line_type is 'key':
        #logger.debug('line is key: {}'.format(line))
        key = line.split(':')[0]
        values_tuple = read_values(all_lines, tuple())
        #logger.debug('values tuple is {}'.format(values_tuple))
        set_dict[key] = values_tuple

    elif line_type in line_special_char_section.values():
        #logger.debug('{} is {}'.format(line, line_type))
        #logger.debug('got to {}'.format(line_type))
        # make the entry if it's the first time this kind of key comes around
        if not line_type in set_dict:
            set_dict[line_type] = tuple()
        # Now read it and append it
        processor_dict = read_recorder_processor(all_lines, dict())
        #logger.debug('read processor {}'.format(processor_dict))
        set_dict[line_type] += (processor_dict, )

    elif line_type is 'end':
        print('Done reading')
        return set_dict

    set_dict = process_top_level(all_lines, set_dict)
    return set_dict


def read_settings_file(file_path: str) -> dict:
    with open(file_path, 'r') as fid:
        file_lines = fid.readlines()
    settings_dict = process_top_level(file_lines, dict())
    return make_chan_info_dict(settings_dict)
