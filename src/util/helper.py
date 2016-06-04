# -*- coding: utf-8 -*-

import os
import sys
import hashlib
import pickle
import progressbar
import threading
import logging


SRC_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..")
APP_PATH = os.path.join(SRC_PATH, "..")
LOG_PATH = os.path.join(APP_PATH, "logs")
CACHE_PATH = os.path.join(APP_PATH, "temp", "cache")


class ExitCode(object):
    SUCCESS = 0
    FAILED = 1


def make_dir_if_not_exists(path):
    paths = [path] if type(path) is not list else path
    for path in paths:
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            elif not os.path.isdir(path):
                error("Invalid path '{0}'. This is NO directory.".format(path))
        except Exception, e:
            error(e)


def error(msg):
    print("ERROR: {0}".format(msg))
    sys.exit(ExitCode.FAILED)


def is_complete_data_set(path):
    required_files = ["Posts.xml", "Tags.xml"]
    available_files = filter(lambda f: os.path.isfile(os.path.join(path, f)), required_files)
    return set(available_files) == set(required_files)


def _cache_file_paths(cached_file_name_prefix):
    cached_tags_file_path = os.path.join(CACHE_PATH, cached_file_name_prefix + "_tags.pickle")
    cached_posts_file_path = os.path.join(CACHE_PATH, cached_file_name_prefix + "_posts.pickle")
    return cached_tags_file_path, cached_posts_file_path


def cache_exists_for_preprocessed_tags_and_posts(cache_file_name_prefix):
    cached_tags_file_path, cached_posts_file_path = _cache_file_paths(cache_file_name_prefix)
    return os.path.exists(cached_tags_file_path) and os.path.exists(cached_posts_file_path)


def write_preprocessed_tags_and_posts_to_cache(cache_file_name_prefix, tags, posts):
    cache_tags_file_path, cache_posts_file_path = _cache_file_paths(cache_file_name_prefix)
    with open(cache_tags_file_path, 'wb') as fp:
        pickle.dump(tags, fp)
    with open(cache_posts_file_path, 'wb') as fp:
        pickle.dump(posts, fp)


def load_preprocessed_tags_and_posts_from_cache(cache_file_name_prefix):
    cache_tags_file_path, cache_posts_file_path = _cache_file_paths(cache_file_name_prefix)
    assert cache_tags_file_path is not None
    assert cache_posts_file_path is not None
    with open(cache_tags_file_path, 'r') as fp:
        cached_preprocessed_tags = pickle.load(fp)
    with open(cache_posts_file_path, 'r') as fp:
        cached_preprocessed_posts = pickle.load(fp)
    # TODO: FIXME RELINK!! tag instances have other id() than those within posts...!!
    return cached_preprocessed_tags, cached_preprocessed_posts


def suggest_random_tags(n_suggested_tags, test_posts, tags):
    from random import randint
    def _random_tag(n_posts_assignments, tags):
        idx = randint(0, (n_posts_assignments-1))
        total_sum = 0
        for t in tags:
            total_sum += t.count
            if idx >= total_sum:
                continue
            #print "{}: {}".format(t.name, t.count)
            return t

    n_posts_assignments = reduce(lambda x,y: x + y, map(lambda t: t.count, tags))
    assert n_posts_assignments > 0
    progress_bar = ProgressBar(len(test_posts))
    for test_post in test_posts:
        suggested_tags = []
        for _ in range(n_suggested_tags):
            suggested_tags += [_random_tag(n_posts_assignments, tags)]
        test_post.tag_set_prediction = suggested_tags
        progress_bar.update()

    progress_bar.finish()


def print_tags_summary(total_num_of_tags, num_filtered_tags):
    print "-"*80
    print "Total number of tags: %d" % total_num_of_tags
    print "Number of remaining tags: %d" % num_filtered_tags
    print "Removed {}% of all tags".format(round(float(total_num_of_tags-num_filtered_tags)/total_num_of_tags*100.0, 2))
    print "-"*80


def print_posts_summary(all_posts, num_of_all_post_assignments, filtered_posts):
    print "-"*80
    print "Total number of posts: %d" % len(all_posts)
    print "Total number of tag assignments: %d" % num_of_all_post_assignments
    print "Number of remaining posts: %d" % len(filtered_posts)
    print "Removed {}% of all posts".format(round(float(len(all_posts)-len(filtered_posts))/len(all_posts)*100.0, 2))

    new_posts_assignments = reduce(lambda x,y: x + y, map(lambda p: len(p.tag_set), filtered_posts))
    print "Removed {}% of all tag assignments".format(round(float(num_of_all_post_assignments-new_posts_assignments)/num_of_all_post_assignments*100.0, 2))
    print "-"*80


def compute_hash_of_file(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def clear_tag_predictions_for_posts(posts):
    for post in posts:
        post.tag_set_prediction = None


class ProgressBar(object):

    def __init__(self, num_of_iterations, output_stream=sys.stdout):
        self._iterations_counter = 0
        self.num_of_iterations = num_of_iterations
        self._output_stream = output_stream
        self.lock = threading.RLock()
        self.finished = False
        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=100)
        self.pbar.start()

    def finish(self):
        with self.lock:
            if self.finished: return
            self.finished = True
            self.pbar.finish()
            assert self.is_full() # Note: reentrant lock! no deadlock can happen here!

    def is_full(self):
        with self.lock:
            return (self._iterations_counter == self.num_of_iterations)

    def update(self, increment=1):
        with self.lock:
            if self.num_of_iterations == None:
                if self._output_stream != None:
                    self._output_stream.write("[WARNING] Number of iterations not set!")
                return

            if self._iterations_counter + increment > self.num_of_iterations:
                if self._output_stream != None:
                    self._output_stream.write("[WARNING] Progress counter overflow!")
                return

            self._iterations_counter = self._iterations_counter + increment
            percentage = float(self._iterations_counter)/float(self.num_of_iterations)*100.0
            self.pbar.update(int(round(percentage)))


# based on: http://stackoverflow.com/a/1336640
# now we patch Python code to add color support to logging.StreamHandler
def add_coloring_to_emit_windows(fn):
    # add methods we need to the class
    def _out_handle(self):
        import ctypes
        return ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE) #@UndefinedVariable
    _ = property(_out_handle)

    def _set_color(self, code):
        import ctypes
        # constants from WinAPI
        self.STD_OUTPUT_HANDLE = -11
        hdl = ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE) #@UndefinedVariable
        ctypes.windll.kernel32.SetConsoleTextAttribute(hdl, code) #@UndefinedVariable

    setattr(logging.StreamHandler, '_set_color', _set_color)

    def new(*args):
        FOREGROUND_BLUE      = 0x0001 # text color contains blue.
        FOREGROUND_GREEN     = 0x0002 # text color contains green.
        FOREGROUND_RED       = 0x0004 # text color contains red.
        FOREGROUND_INTENSITY = 0x0008 # text color is intensified.
        FOREGROUND_WHITE     = FOREGROUND_BLUE|FOREGROUND_GREEN |FOREGROUND_RED
        # winbase.h
        STD_INPUT_HANDLE = -10
        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12

        # wincon.h
        FOREGROUND_BLACK     = 0x0000
        FOREGROUND_BLUE      = 0x0001
        FOREGROUND_GREEN     = 0x0002
        FOREGROUND_CYAN      = 0x0003
        FOREGROUND_RED       = 0x0004
        FOREGROUND_MAGENTA   = 0x0005
        FOREGROUND_YELLOW    = 0x0006
        FOREGROUND_GREY      = 0x0007
        FOREGROUND_INTENSITY = 0x0008 # foreground color is intensified.

        BACKGROUND_BLACK     = 0x0000
        BACKGROUND_BLUE      = 0x0010
        BACKGROUND_GREEN     = 0x0020
        BACKGROUND_CYAN      = 0x0030
        BACKGROUND_RED       = 0x0040
        BACKGROUND_MAGENTA   = 0x0050
        BACKGROUND_YELLOW    = 0x0060
        BACKGROUND_GREY      = 0x0070
        BACKGROUND_INTENSITY = 0x0080 # background color is intensified.     

        levelno = args[1].levelno
        if(levelno>=50):
            color = BACKGROUND_YELLOW | FOREGROUND_RED | FOREGROUND_INTENSITY | BACKGROUND_INTENSITY 
        elif(levelno>=40):
            color = FOREGROUND_RED | FOREGROUND_INTENSITY
        elif(levelno>=30):
            color = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
        elif(levelno>=20):
            color = FOREGROUND_GREEN
        elif(levelno>=10):
            color = FOREGROUND_MAGENTA
        else:
            color =  FOREGROUND_WHITE
        args[0]._set_color(color)

        ret = fn(*args)
        args[0]._set_color( FOREGROUND_WHITE )
        #print "after"
        return ret
    return new

def add_coloring_to_emit_ansi(fn):
    # add methods we need to the class
    def new(*args):
        levelno = args[1].levelno
        if levelno >= 50:
            color = '\x1b[31m' # red
        elif(levelno>=40):
            color = '\x1b[31m' # red
        elif(levelno>=30):
            color = '\x1b[33m' # yellow
        elif(levelno>=20):
            color = '\x1b[32m' # green 
        elif(levelno>=10):
            color = '\x1b[35m' # pink
        else:
            color = '\x1b[0m' # normal
        args[1].msg = color + args[1].msg +  '\x1b[0m'  # normal
        #print "after"
        return fn(*args)
    return new
