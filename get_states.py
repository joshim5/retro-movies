import retro
import os
import cv2

from baselines.common.atari_wrappers import WarpFrame, FrameStack

#e.g. movie_path = 'human/SonicAndKnuckles3-Genesis/contest/SonicAndKnuckles3-Genesis-DeathEggZone.Act2-0000.bk2'

def proc_obs(o, warpframe=None):
    option = 'baselines'

    assert o.shape == (224, 320, 3), 'expected raw images'
    if option == 'manual_bw':
      o = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
      o = cv2.resize(o, (84,84), interpolation=cv2.INTER_AREA)  # 4x smaller
      #80,56
      return o[:,:,None]
    if option == 'manual_color':
      o = cv2.resize(o, (84,84), interpolation=cv2.INTER_AREA)  # 4x smaller
      #80,56
      return o[:,:,None]
    if option == 'baselines':
      return warpframe.observation(o)

def process_movie(movie_path, viewer, step_after_traj = False):
    movie = retro.Movie(movie_path)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
    env.initial_state = movie.get_state()
    warpframe = WarpFrame(env)
    env.reset()

    print('stepping movie')
    num_steps = 0

    while movie.step():
        num_steps += 1
        keys = []
        for i in range(len(env.buttons)):
            keys.append(movie.get_key(i, 0))
        _obs, _rew, _done, _info = env.step(keys)
        viewer.imshow(proc_obs(_obs, warpframe))
        saved_state = env.em.get_state()

    print(num_steps, 'steps exist in the movie.')

    if step_after_traj:
      print('stepping environment started at final state of movie')
      env.initial_state = saved_state
      env.reset()
      while True:
          env.render()
          env.step(env.action_space.sample())

    return num_steps

if __name__ == '__main__':
  # Counter metrics
  total_num_steps = 0
  total_num_videos = 0

  # Custom control of the rendering
  from gym.envs.classic_control.rendering import SimpleImageViewer
  viewer = SimpleImageViewer()

  for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".bk2")]:
        movie_path = os.path.join(dirpath, filename)
        print('processing', filename)
        total_num_steps += process_movie(movie_path, viewer)
        total_num_videos += 1

  print('total_num_videos =', total_num_videos, 'total_num_steps =', total_num_steps)
