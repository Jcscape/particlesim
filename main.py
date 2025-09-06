from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pygame
from .engine import Engine, Params
from .colors import hsv_to_rgb

W, H = 1280, 720

def load_preset(path: str, prm: Params) -> Params:
    with open(path, 'r') as f:
        d = json.load(f)
    prm.n = d.get('num_particles', prm.n)
    prm.types = d.get('num_types', prm.types)
    prm.radius = float(d.get('radius', prm.radius))
    prm.fric = float(d.get('friction', prm.fric))
    prm.dt = float(d.get('dt', prm.dt))
    prm.bias = float(d.get('force_bias', prm.bias))
    prm.frange = float(d.get('force_range', prm.frange))
    prm.fmul = float(d.get('force_mul', prm.fmul))
    prm.food = float(d.get('food_density', prm.food))
    prm.reg = float(d.get('regrow_rate', prm.reg))
    prm.birth = float(d.get('birth_energy', prm.birth))
    prm.maxage = int(d.get('max_age', prm.maxage))
    return prm

def draw(engine: Engine, screen, trails=True, show_field=False):
    # Trails: draw semi-transparent rect
    if trails:
        s = pygame.Surface((W,H))
        s.set_alpha(56)
        s.fill((11,13,18))
        screen.blit(s, (0,0))
    else:
        screen.fill((11,13,18))

    # optional: draw food field
    if show_field and (pygame.time.get_ticks()//80)%2==0:
        cell = 12
        for y in range(engine.fields.rows):
            for x in range(engine.fields.cols):
                f = engine.fields.food[y,x]
                if f>0.05:
                    alpha = int(25*f)
                    if alpha>0:
                        pygame.draw.rect(screen, (90,150,255,alpha), pygame.Rect(x*cell, y*cell, cell, cell), 0)

    # draw agents
    base_r = max(3, int(engine.prm.radius*0.16))
    for a in engine.A:
        # glow ring
        pygame.draw.circle(screen, (255,255,255,18), (int(a.x), int(a.y)), int(base_r*1.8), 0)
        col = hsv_to_rgb((a.t/ max(1, engine.prm.types) + a.hue*0.2)%1.0, 0.7, 0.95)
        pygame.draw.circle(screen, col, (int(a.x), int(a.y)), base_r, 0)

def main(argv=None):
    ap = argparse.ArgumentParser(prog='particlesim', description='Living Patterns — Python')
    ap.add_argument('--preset', type=int, default=1, help='Preset number 1..5')
    ap.add_argument('--n', type=int, help='Override particle count')
    ap.add_argument('--types', type=int, help='Override type count')
    ap.add_argument('--frames', type=int, default=0, help='If >0, run for N frames then exit (useful for recording)')
    ap.add_argument('--record', type=str, help='Output GIF filename (requires imageio)')
    ap.add_argument('--width', type=int, default=W)
    ap.add_argument('--height', type=int, default=H)
    args = ap.parse_args(argv)

    global W,H
    W, H = args.width, args.height
    pygame.init()
    screen = pygame.display.set_mode((W,H), pygame.SRCALPHA, 0)
    pygame.display.set_caption('ParticleSim — Living Patterns')

    prm = Params()
    preset_path = os.path.join(os.path.dirname(__file__), '..', 'presets', f'{args.preset}.json')
    if os.path.exists(preset_path):
        prm = load_preset(preset_path, prm)

    if args.n: prm.n = args.n
    if args.types: prm.types = args.types

    eng = Engine(W,H, prm, seed=int(time.time()))
    clock = pygame.time.Clock()

    paused = False
    trails = True
    show_field = False
    frame = 0
    recorder = None
    if args.record:
        import imageio
        recorder = imageio.get_writer(args.record, fps=30)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_p: paused = not paused
                elif ev.key == pygame.K_s and paused: # step
                    eng.step()
                elif ev.key == pygame.K_r:
                    eng.reset_agents()
                elif ev.key == pygame.K_SPACE:
                    eng.regen_rules()
                elif ev.key == pygame.K_g:
                    trails = not trails
                elif ev.key == pygame.K_h:
                    print('Keys: P pause, S step, R reset, Space regen rules, G toggle trails, F screenshot, digits 1..5 load presets, F8 toggle field')
                elif ev.key == pygame.K_F8:
                    show_field = not show_field
                elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    idx = {pygame.K_1:1, pygame.K_2:2, pygame.K_3:3, pygame.K_4:4, pygame.K_5:5}[ev.key]
                    prm = load_preset(os.path.join(os.path.dirname(__file__), '..','presets', f'{idx}.json'), eng.prm)
                    eng.prm = prm
                    eng.set_counts(prm.n, prm.types)
                elif ev.key == pygame.K_f:
                    fname = os.path.join(os.path.dirname(__file__), '..', 'assets', f'shot_{int(time.time())}.png')
                    pygame.image.save(screen, fname)
                    print('Saved', fname)

        if not paused:
            eng.step()

        draw(eng, screen, trails=trails, show_field=show_field)
        pygame.display.flip()

        if recorder:
            # read pixels
            data = pygame.surfarray.array3d(screen).swapaxes(0,1)
            recorder.append_data(data)

        clock.tick(60)
        frame += 1
        if args.frames>0 and frame>=args.frames:
            running=False

    if recorder:
        recorder.close()
    pygame.quit()

if __name__ == '__main__':
    main()
