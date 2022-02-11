from abc import abstractmethod, ABC
from numpy import cos, sin, tan, sqrt, uint8, array, zeros, dot, abs
from typing import Callable
from PIL import Image
import time
from threading import Thread


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class SDF(ABC):
    def __init__(self, x=0, y=0, z=0, scale_x=1.0, scale_y=1.0, scale_z=1.0, a=0, b=0):
        self.x = x
        self.y = y
        self.z = z
        self.scale_x = 1 / scale_x
        self.scale_y = 1 / scale_y
        self.scale_z = 1 / scale_z
        self.a = a
        self.b = b
        self.rotation_matrix = array([
            [cos(a) * cos(b), -cos(a) * sin(b), sin(a)],
            [sin(b), cos(b), 0],
            [-cos(b) * sin(a), sin(a) * sin(b), cos(a)]
        ])

    def get_reverce_value(self, x, y, z):
        cam_coordinates = dot(self.rotation_matrix, array([
            [x[0] + self.x * self.scale_x], [y[0] + self.y * self.scale_y], [z[0] + self.z * self.scale_z],
        ]))
        return self.reverce_formula(
            x=cam_coordinates[0],
            y=cam_coordinates[1],
            z=cam_coordinates[2],
        )

    def get_value(self, x, y, z):
        return self.formula(
            x=x + self.x * self.scale_x,
            y=y + self.y * self.scale_y,
            z=z + self.z * self.scale_z,
        )

    def get_value_extended(self, x, y, z):
        cam_coordinates = dot(self.rotation_matrix, array([
            [x[0] + self.x * self.scale_x], [y[0] + self.y * self.scale_y], [z[0] + self.z * self.scale_z],
        ]))
        return self.formula(
            x=cam_coordinates[0],
            y=cam_coordinates[1],
            z=cam_coordinates[2],
        )

    @abstractmethod
    def formula(self, x, y, z):
        pass


class Sphere(SDF):
    def __init__(self, x=0, y=0, z=0, scale_x=1.0, scale_y=1.0, scale_z=1.0, a=0, b=0, radius=1.0):
        self.r = radius
        super().__init__(x, y, z, scale_x, scale_y, scale_z, a, b)

    def reverce_formula(self, x, y, z):
        return sqrt(x ** 2 + y ** 2 + z ** 2) + self.r

    def formula(self, x, y, z):
        return sqrt(x ** 2 + y ** 2 + z ** 2) - self.r


class Cube(SDF):
    def __init__(self, x=0, y=0, z=0, scale_x=1.0, scale_y=1.0, scale_z=1.0, a=0, b=0, side=1.0):
        self.s = side
        super().__init__(x, y, z, scale_x, scale_y, scale_z, a, b)

    def reverce_formula(self, x, y, z):
        return max(abs(x), abs(y), abs(z)) + self.s / 2

    def formula(self, x, y, z):
        return max(abs(x), abs(y), abs(z)) - self.s / 2


class Tor(SDF):
    def __init__(self, x=0, y=0, z=0, scale_x=1.0, scale_y=1.0, scale_z=1.0, a=0, b=0, r=0.125, B_R=0.75):
        self.r = r
        self.B_R = B_R
        super().__init__(x, y, z, scale_x, scale_y, scale_z, a, b)

    def reverce_formula(self, x, y, z):
        return sqrt((sqrt(x ** 2 + z ** 2) - self.B_R) ** 2 + y ** 2) + self.r

    def formula(self, x, y, z):
        return sqrt((sqrt(x ** 2 + z ** 2) - self.B_R) ** 2 + y ** 2) - self.r


class Plane(SDF):
    def __init__(self, x=0, y=0, z=0, scale_x=1.0, scale_y=1.0, scale_z=1.0, a=0, b=0):
        super().__init__(x, y, z, scale_x, scale_y, scale_z, a, b)

    def reverce_formula(self, x, y, z):
        return y

    def formula(self, x, y, z):
        return y


class obj_plus_obj:
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2

    def get_reverce_value(self, x, y, z):
        return max(self.obj1.get_reverce_value(x, y, z), self.obj2.get_reverce_value(x, y, z))

    def get_value(self, x, y, z):
        return min(self.obj1.get_value(x, y, z), self.obj2.get_value(x, y, z))

    def get_value_extended(self, x, y, z):
        return min(self.obj1.get_value_extended(x, y, z), self.obj2.get_value_extended(x, y, z))


class obj_minus_obj:
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2

    def get_reverce_value(self, x, y, z):
        return max(self.obj1.get_reverce_value(x, y, z), -self.obj2.get_reverce_value(x, y, z))

    def get_value(self, x, y, z):
        return max(self.obj1.get_value(x, y, z), -self.obj2.get_value(x, y, z))

    def get_value_extended(self, x, y, z):
        return max(self.obj1.get_value_extended(x, y, z), -self.obj2.get_value_extended(x, y, z))


class render:
    def __init__(self, w, h, a, b, cam_dist, max_dist,
                 sdf: Callable[[float, float, float], float] = lambda x, y, z: 0.0):
        self.w = w
        self.h = h
        self.a = a
        self.b = b
        self.cam_dist = cam_dist
        self.cam_angel = 39
        self.light_pos = (-5, 5, 0)  # -5, 5, 0
        self.rotation_matrix = array([
            [cos(a) * cos(b), -cos(a) * sin(b), sin(a)],
            [sin(b), cos(b), 0],
            [-cos(b) * sin(a), sin(a) * sin(b), cos(a)]
        ])
        self.cam_coordinates = dot(self.rotation_matrix, array([
            [cam_dist], [0], [0],
        ]))
        # print(self.cam_coordinates)
        # print(self.rotation_matrix[0][0] * cam_dist, self.rotation_matrix[1][0] * cam_dist, self.rotation_matrix[2][0] * cam_dist)
        self.max_dist = max_dist(
            self.cam_coordinates[0],
            self.cam_coordinates[1],
            self.cam_coordinates[2]
        )
        self.sdf = sdf
        self.orig_dist = self.sdf(
            self.cam_coordinates[0],
            self.cam_coordinates[1],
            self.cam_coordinates[2]
        )[0]
        self.pos0 = dot(self.rotation_matrix, array([
            [-1], [0], [0],
        ]))
        self.pixel_size = 2 * (tan(self.cam_angel / 2) / h)
        self.u = dot(self.rotation_matrix, array([
            [0], [0], [1],
        ])) * self.pixel_size
        self.v = dot(self.rotation_matrix, array([
            [0], [1], [0],
        ])) * self.pixel_size

    def raydir(self, x, y):
        ray_x = self.pos0[0] + (x * self.u[0]) + (y * self.v[0])
        ray_y = self.pos0[1] + (x * self.u[1]) + (y * self.v[1])
        ray_z = self.pos0[2] + (x * self.u[2]) + (y * self.v[2])
        return ray_x, ray_y, ray_z

    def ray(self, dist, dir):
        ray_x = self.cam_coordinates[0] + (dist * dir[0])
        ray_y = self.cam_coordinates[1] + (dist * dir[1])
        ray_z = self.cam_coordinates[2] + (dist * dir[2])
        return ray_x, ray_y, ray_z

    def light(self, dist, start_pos, dir):
        light_x = start_pos[0] + dist * dir[0]
        light_y = start_pos[1] + dist * dir[1]
        light_z = start_pos[2] + dist * dir[2]
        return light_x, light_y, light_z

    def raymarch_light(self, start_pos, ray_dir, k):
        dist_to_light = sqrt((self.light_pos[0] - start_pos[0]) ** 2 + (self.light_pos[1] - start_pos[1]) ** 2 + (
                    self.light_pos[2] - start_pos[2]) ** 2)

        dist = self.sdf(start_pos[0], start_pos[1], start_pos[2])

        ray_pos = light(dist, start_pos, ray_dir)
        delta = self.sdf(ray_pos[0], ray_pos[1], ray_pos[2])
        dist += delta

        ray_pos = light(dist, start_pos, ray_dir)
        delta = self.sdf(ray_pos[0], ray_pos[1], ray_pos[2])
        dist += delta
        for _ in range(100):
            if delta < 0.001:
                return 0
            if dist >= dist_to_light:
                return k
            ray_pos = light(dist, start_pos, ray_dir)
            delta = self.sdf(ray_pos[0], ray_pos[1], ray_pos[2])
        return k

    def stripe_rendering(self, h1, h2):
        h = round(self.h / 2)
        w = round(self.w / 2)
        res = zeros((abs(h2 - h1), self.w))
        for i in range(h1, h2):
            for l in range(-w, w):
                ray_dir = self.raydir(l, i)
                dist = 0
                ray_pos = self.ray(self.orig_dist, ray_dir)
                delta = self.sdf(ray_pos[0], ray_pos[1], ray_pos[2])
                for _ in range(100):
                    if delta < 0.001:
                        ray_pos = self.ray(dist + delta, ray_dir)
                        light_dist = sqrt(
                            (ray_pos[0] - self.light_pos[0]) ** 2 +
                            (ray_pos[1] - self.light_pos[1]) ** 2 +
                            (ray_pos[2] - self.light_pos[2]) ** 2
                        )
                        light_dir = (
                            ((ray_pos[0] - self.light_pos[0]) / light_dist)[0],
                            ((ray_pos[1] - self.light_pos[1]) / light_dist)[0],
                            ((ray_pos[2] - self.light_pos[2]) / light_dist)[0]
                        )
                        sdf_value = self.sdf(ray_pos[0], ray_pos[1], ray_pos[2])
                        x = sdf_value - self.sdf(ray_pos[0] - 0.0001, ray_pos[1], ray_pos[2])
                        y = sdf_value - self.sdf(ray_pos[0], ray_pos[1] - 0.0001, ray_pos[2])
                        z = sdf_value - self.sdf(ray_pos[0], ray_pos[1], ray_pos[2] - 0.0001)
                        just_dist = sqrt(x ** 2 + y ** 2 + z ** 2)
                        normal = [
                            (x / just_dist)[0],
                            (y / just_dist)[0],
                            (z / just_dist)[0]
                        ]
                        # print(normal)
                        # print(light_dir)
                        dist = (abs(dot(normal, light_dir)) * 170 + 60)
                        break
                    if delta > self.max_dist:
                        dist = 255
                        break
                    dist += delta
                    ray_pos = self.ray(dist, ray_dir)
                    delta = self.sdf(ray_pos[0], ray_pos[1], ray_pos[2])
                res[i - h1, l + w] = dist
        return res

    def rendering(self):
        Image = zeros((self.h, self.w))
        h2 = round(self.h / 2)
        h4 = round(h2 / 2)
        h6 = round(h4 * 1.5)
        h8 = round(h4 / 2)

        t = time.time()
        steep = 1
        self.stripe_rendering(-h8 - steep, -h8 + steep)
        self.stripe_rendering(-h4 - steep, -h4 + steep)
        self.stripe_rendering(-h6 - steep, -h6 + steep)
        self.stripe_rendering(-steep, steep)
        self.stripe_rendering(h8 - steep, h8 + steep)
        self.stripe_rendering(h4 - steep, h4 + steep)
        self.stripe_rendering(h6 - steep, h6 + steep)
        sec = ((time.time() - t) / (steep * 14)) * self.h
        if sec >= 60:
            if sec >= 3600:
                print(int(sec / 3600), "hours", int((sec % 3600) / 60), "min", round((sec % 3600) % 60), "sec")
            else:
                print(int(sec / 60), "min", round(sec % 60), "sec")
        else:
            print(sec, "sec")
        print(time.strftime("Will done in %H:%M", time.localtime(time.time() + sec)))
        # print(round(128/((time.time() - t)/(steep*14))), "pps")

        m = 16
        threads = [ThreadWithReturnValue(target=self.stripe_rendering, args=[i, i+round(self.h/m)]) for i in range(-h2, h2-round(self.h/m)+1, round(self.h/m))]
        for i in range(len(threads)):
            threads[i].start()
        for i in range(len(threads)):
            Image[round(self.h/m)*i:round(self.h/m)*(i+1)] = threads[i].join()
        return Image


def main():
    sphere = Sphere(x=-0.9, y=0, z=0.5, radius=0.75)
    cube = Cube(x=1, y=0.5, z=0, side=1.5, a=0.5)
    sphere_to_cut = Sphere(x=1, y=0.5, z=0)
    tor = Tor(x=0.3, y=0.75, z=-0.7, r=0.125, B_R=0.5)

    w = 1024*16
    h = 1024*16

    r = render(
        w,
        h,
        -0.75,
        -0.45,
        3,
        obj_minus_obj(
            cube,
            sphere_to_cut
        ).get_reverce_value,
        obj_minus_obj(
            cube,
            sphere_to_cut
        ).get_value_extended
    )

    im = r.rendering()
    im = Image.fromarray(im.astype(uint8), 'L')
    im.save("raymarched.png")
    im.show()


if __name__ == "__main__":
    main()
