import pandas as pd

from clarite import io, modify


def test_save_and_load(plantTraits, capfd):
    io.save(plantTraits, 'test_output/saved.txt')
    loaded = io.load_data('test_output/saved.txt')
    pd.testing.assert_frame_equal(loaded, plantTraits)
    return


def test_save_with_cat(plantTraits, capfd):
    df = plantTraits
    df['begflow'] = modify.make_categorical(df['begflow'])
    df['piq'] = modify.make_binary(df['piq'])

    io.save(df, 'test_output/saved.txt')
    loaded = io.load_data('test_output/saved.txt')
    pd.testing.assert_frame_equal(loaded, plantTraits)
    return
