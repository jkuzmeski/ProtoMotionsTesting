import carb
import numpy as np
from pxr import Gf, Sdf

class PerspectiveViewer(object):
    def __init__(self):
        self.viewport_api = None
        self.get_viewport_api()
        
        # Only try to change render settings if we successfully found a viewport
        if self.viewport_api:
            self.disable_advanced_rendering()

    def disable_advanced_rendering(self):
        # A check to prevent crashing if the viewport is not available
        if not self.viewport_api:
            return
            
        stage = self.viewport_api.stage
        render_settings_path = "/Render/RenderProduct/RenderSettings"
        
        render_settings = stage.GetPrimAtPath(render_settings_path)
        if not render_settings.IsValid():
            render_settings = stage.DefinePrim(render_settings_path, "RenderSettings")

        render_settings.CreateAttribute("rtx:raytracing:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        render_settings.CreateAttribute("rtx:pathtracing:gi:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        render_settings.CreateAttribute("rtx:ambientOcclusion:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        render_settings.CreateAttribute("rtx:dof:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        render_settings.CreateAttribute("rtx:pathtracing:maxBounces", Sdf.ValueTypeNames.Int).Set(1)
        render_settings.CreateAttribute("rtx:pathtracing:maxSamples", Sdf.ValueTypeNames.Int).Set(16)

        stage.SetEditTarget(stage.GetSessionLayer())


    def get_viewport_api(self):
        if self.viewport_api is None:
            try:
                from omni.kit.viewport.utility import get_active_viewport
                self.viewport_api = get_active_viewport()
            except ImportError:
                carb.log_warn(
                    "omni.kit.viewport.utility needs to be enabled before using this function"
                )

            if self.viewport_api is None:
                carb.log_warn("Could not get active viewport. Camera view settings will be unavailable.")

    def get_camera_state(self):
        self.get_viewport_api()
        
        # Return a default value if the viewport is not available
        if not self.viewport_api:
            return 0.0, 0.0, 0.0

        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        prim = self.viewport_api.stage.GetPrimAtPath("/OmniverseKit_Persp")

        coi_prop = prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            prim.CreateAttribute(
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))
        camera_state = ViewportCameraState("/OmniverseKit_Persp", self.viewport_api)
        camera_position = camera_state.position_world
        return camera_position[0], camera_position[1], camera_position[2]

    def set_camera_view(self, eye: np.array, target: np.array):
        self.get_viewport_api()
        
        # Do nothing if the viewport is not available
        if not self.viewport_api:
            return

        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        camera_position = np.asarray(eye, dtype=np.double)
        camera_target = np.asarray(target, dtype=np.double)
        prim = self.viewport_api.stage.GetPrimAtPath("/OmniverseKit_Persp")

        coi_prop = prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            prim.CreateAttribute(
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))
        camera_state = ViewportCameraState("/OmniverseKit_Persp", self.viewport_api)
        camera_state.set_position_world(
            Gf.Vec3d(camera_position[0], camera_position[1], camera_position[2]), True
        )
        camera_state.set_target_world(
            Gf.Vec3d(camera_target[0], camera_target[1], camera_target[2]), True
        )