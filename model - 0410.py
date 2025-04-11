import mesa
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
import random
import os
import math
from datetime import datetime

# Check Mesa version
print(f"Mesa version used: {mesa.__version__}")

# Define Agent class
class ResidentAgent(mesa.Agent):
    def __init__(self, unique_id, model, weightForest=1, weightService=1):
        mesa.Agent.__init__(self, model)
        self.unique_id = unique_id
        self.weightForest = weightForest
        self.weightService = weightService
        self.utility = 0
        
    def step(self):
        # Agents don't need to execute any steps in this model
        pass

class ServiceAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        mesa.Agent.__init__(self, model)
        self.unique_id = unique_id
        
    def step(self):
        # Agents don't need to execute any steps in this model
        pass

# Define model (modifiable grid size and spatial resolution)
class UrbanGrowthModel(mesa.Model):
    def __init__(self, 
                 width=1000, 
                 height=1000, 
                 spatial_resolution=1,
                 forest_gradient_service=1, 
                 market_knowledge=16, 
                 allowed_outside_houses=100, 
                 city_radius=25, 
                 smoothness=15, 
                 change_forest_nearby=False,
                 export_path="./data/",
                 experiment_type="baseline"):  # Add experiment type parameter
        
        super().__init__()
        
        # Initialize model parameters
        self.width = width
        self.height = height
        self.spatial_resolution = spatial_resolution
        self.forest_gradient_service = forest_gradient_service
        self.market_knowledge = market_knowledge
        self.allowed_outside_houses = allowed_outside_houses
        self.city_radius = city_radius
        self.smoothness = smoothness
        self.change_forest_nearby = change_forest_nearby
        self.export_path = export_path
        self.experiment_type = experiment_type  # Save experiment type
        self.running = True
        self.steps = 0
        
        # Ensure output directory exists
        os.makedirs(export_path, exist_ok=True)
        
        # Create model timestamp as run ID
        self.run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Initialize grid - using MultiGrid
        self.grid = MultiGrid(width, height, False)
        
        # Initialize model state variables
        self.counter = 0
        self.selx = 0
        self.sely = 0
        self.residentsperstep = 10
        self.residentsperservice = 100
        self.avg_forest_quality = 0
        self.avg_distance_to_service = 0
        self.next_id_value = 0
        
        # Create data collector, add experiment type field
        self.datacollector = DataCollector(
            model_reporters={
                "Residents": lambda m: len([a for a in m.agents if isinstance(a, ResidentAgent)]),
                "Services": lambda m: len([a for a in m.agents if isinstance(a, ServiceAgent)]),
                "Outside_Residents": lambda m: self.count_outside_residents(),
                "Forest_Coverage": lambda m: self.calculate_forest_coverage(),
                "Experiment_Type": lambda m: self.experiment_type,  # Add experiment type field
                "Spatial_Resolution": lambda m: self.spatial_resolution  # Add spatial resolution field
            }
        )
        
        # Initialize grid forest values and service distance values
        self.forest = np.zeros((width, height))
        self.sddist = np.full((width, height), np.inf)
        
        # Set initial forest state
        self.setup_forest()
        
        # Create initial service center
        self.setup_service()
        
        # Collect initial data
        self.datacollector.collect(self)
        
        # Initialize CSV file, pass experiment type
        self.init_csv_file(experiment_type)
        
        # Initialize grid data record file, pass experiment type
        self.init_grid_data_file(experiment_type)
    
    def next_id(self):
        """Method to provide unique ID"""
        self.next_id_value += 1
        return self.next_id_value
    
    def setup_forest(self):
        # Initialize forest random values
        max_dist = math.sqrt(self.width**2 + self.height**2)
        for x in range(self.width):
            for y in range(self.height):
                self.forest[x, y] = random.uniform(0, max_dist)
        
        # Smooth forest values
        for _ in range(self.smoothness):
            new_forest = np.copy(self.forest)
            for x in range(self.width):
                for y in range(self.height):
                    neighbors_sum = 0
                    count = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                neighbors_sum += self.forest[nx, ny]
                                count += 1
                    new_forest[x, y] = neighbors_sum / count
            self.forest = new_forest
    
    def setup_service(self):
        # Create initial service center in the middle location
        mid_x, mid_y = self.width // 2, self.height // 2
        service_agent = ServiceAgent(self.next_id(), self)
        self.grid.place_agent(service_agent, (mid_x, mid_y))
        
        # Update service distance matrix
        self.update_service_distances()
    
    def update_service_distances(self):
        # Get all service positions
        service_positions = []
        for agent in self.agents:
            if isinstance(agent, ServiceAgent):
                x, y = agent.pos
                service_positions.append((x, y))
        
        # Update distance to nearest service for each cell
        for x in range(self.width):
            for y in range(self.height):
                min_dist = float('inf')
                for sx, sy in service_positions:
                    dist = math.sqrt((x - sx)**2 + (y - sy)**2)
                    min_dist = min(min_dist, dist)
                self.sddist[x, y] = min_dist
    
    def init_csv_file(self, experiment_type="baseline"):
        """Create CSV file and write header row, use unified naming convention including experiment type"""
        # Unified naming format: abm_data_{spatial_resolution}x{spatial_resolution}_{experiment_type}.csv
        filename = f"{self.export_path}abm_data_{self.spatial_resolution}x{self.spatial_resolution}_{experiment_type}.csv"
        
        headers = ["run", "tick", "Plot_Houses", "Development_Beyond_City_Boundary", 
                  "Plot_Forest", "forest_gradient_service", "market_knowledge", 
                  "allowed_outside_houses", "city_radius", "smoothness", "change_forest_nearby", 
                  "experiment_type", "spatial_resolution"]  # Add experiment type and spatial resolution fields
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # If file doesn't exist, create and write headers
        if not os.path.exists(filename):
            df = pd.DataFrame(columns=headers)
            df.to_csv(filename, index=False)
            print(f"Created new data file: {filename}")
    
    def init_grid_data_file(self, experiment_type="baseline"):
        """Create grid data record file, use unified naming convention including experiment type"""
        # Unified naming format: grid_data_{run_ID}_{spatial_resolution}x{spatial_resolution}_{experiment_type}.csv
        filename = f"{self.export_path}grid_data_{self.run_id}_{self.spatial_resolution}x{self.spatial_resolution}_{experiment_type}.csv"
        
        headers = ["run", "tick", "x", "y", "forest_quality", "service_distance", "has_resident", 
                   "has_service", "experiment_type", "spatial_resolution"]  # Add experiment type and spatial resolution fields
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Create file and write header row
        df = pd.DataFrame(columns=headers)
        df.to_csv(filename, index=False)
        print(f"Created new grid data file: {filename}")
    
    def export_data_tick(self, experiment_type="baseline"):
        """Export data for each step, experiment type used for file naming"""
        filename = f"{self.export_path}abm_data_{self.spatial_resolution}x{self.spatial_resolution}_{experiment_type}.csv"
        
        # Calculate grid-based forest coverage
        forest_coverage = self.calculate_forest_coverage()
        
        # Calculate residents outside city boundary
        outside_residents = self.count_outside_residents()
        
        # Get total resident count
        residents_count = len([a for a in self.agents if isinstance(a, ResidentAgent)])
        
        # Collect current state data
        data = {
            "run": self.run_id,
            "tick": self.steps,
            "Plot_Houses": residents_count,
            "Development_Beyond_City_Boundary": outside_residents,
            "Plot_Forest": forest_coverage,
            "forest_gradient_service": self.forest_gradient_service,
            "market_knowledge": self.market_knowledge,
            "allowed_outside_houses": self.allowed_outside_houses,
            "city_radius": self.city_radius,
            "smoothness": self.smoothness,
            "change_forest_nearby": "TRUE" if self.change_forest_nearby else "FALSE",
            "experiment_type": experiment_type,
            "spatial_resolution": self.spatial_resolution
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Write data to CSV file (append mode)
        df = pd.DataFrame([data])
        
        # Check if file exists, if not write headers
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as f:
            df.to_csv(f, header=not file_exists, index=False)
    
    def export_grid_data(self, experiment_type="baseline"):
        """Export forest quality and service distance data for each grid cell, add experiment type identifier"""
        # Unified naming format
        filename = f"{self.export_path}grid_data_{self.run_id}_{self.spatial_resolution}x{self.spatial_resolution}_{experiment_type}.csv"
        
        # Prepare data list
        data_rows = []
        
        # Get resident and service positions
        resident_positions = set()
        service_positions = set()
        
        for agent in self.agents:
            if isinstance(agent, ResidentAgent):
                resident_positions.add(agent.pos)
            elif isinstance(agent, ServiceAgent):
                service_positions.add(agent.pos)
        
        # Collect data for all grid cells
        for x in range(self.width):
            for y in range(self.height):
                has_resident = (x, y) in resident_positions
                has_service = (x, y) in service_positions
                
                data_rows.append({
                    "run": self.run_id,
                    "tick": self.steps,
                    "x": x,
                    "y": y,
                    "forest_quality": self.forest[x, y],
                    "service_distance": self.sddist[x, y],
                    "has_resident": "TRUE" if has_resident else "FALSE",
                    "has_service": "TRUE" if has_service else "FALSE",
                    "experiment_type": experiment_type,  # Add experiment type
                    "spatial_resolution": self.spatial_resolution  # Add spatial resolution
                })
        
        # Write data to CSV file (append mode)
        df = pd.DataFrame(data_rows)
        
        with open(filename, 'a', newline='') as f:
            df.to_csv(f, header=False, index=False)
        
        print(f"Exported grid data: tick={self.steps}, records={len(data_rows)}, experiment_type={experiment_type}")
    
    def count_outside_residents(self):
        # Count residents outside city radius
        mid_x, mid_y = self.width // 2, self.height // 2
        count = 0
        
        for agent in self.agents:
            if isinstance(agent, ResidentAgent):
                x, y = agent.pos
                dist = math.sqrt((x - mid_x)**2 + (y - mid_y)**2)
                if dist >= self.city_radius:
                    count += 1
        
        return count
    
    def calculate_forest_coverage(self):
        """Calculate grid-based forest coverage median, considering spatial resolution"""
        forest_values = []
        
        # Divide grid according to spatial resolution
        grid_width = max(1, self.width // self.spatial_resolution)
        grid_height = max(1, self.height // self.spatial_resolution)
        
        for grid_x in range(grid_width):
            for grid_y in range(grid_height):
                # Calculate actual coordinate range
                x_start = grid_x * self.spatial_resolution
                y_start = grid_y * self.spatial_resolution
                x_end = min(x_start + self.spatial_resolution, self.width)
                y_end = min(y_start + self.spatial_resolution, self.height)
                
                # Collect forest values in this grid
                grid_forest_values = []
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        grid_forest_values.append(self.forest[x, y])
                
                if grid_forest_values:
                    forest_values.append(np.median(grid_forest_values))
        
        # Return median of all grid medians
        if forest_values:
            return np.median(forest_values)
        return 0
    
    def calculate_metrics(self):
        # Calculate system metrics - grid-based medians
        residents = [agent for agent in self.agents if isinstance(agent, ResidentAgent)]
        
        # Set default values if no residents
        if not residents:
            self.avg_forest_quality = 0
            self.avg_distance_to_service = 0
            return
        
        # Divide grid according to spatial resolution
        grid_width = max(1, self.width // self.spatial_resolution)
        grid_height = max(1, self.height // self.spatial_resolution)
        
        # Find grids with residents
        forest_values = []
        distance_values = []
        
        for grid_x in range(grid_width):
            for grid_y in range(grid_height):
                x_start = grid_x * self.spatial_resolution
                y_start = grid_y * self.spatial_resolution
                x_end = min(x_start + self.spatial_resolution, self.width)
                y_end = min(y_start + self.spatial_resolution, self.height)
                
                # Check if this grid has residents
                has_residents = False
                for agent in residents:
                    x, y = agent.pos
                    if x_start <= x < x_end and y_start <= y < y_end:
                        has_residents = True
                        break
                
                if has_residents:
                    # Collect forest values in this grid
                    grid_forest = []
                    grid_distances = []
                    
                    for x in range(x_start, x_end):
                        for y in range(y_start, y_end):
                            grid_forest.append(self.forest[x, y])
                            grid_distances.append(self.sddist[x, y])
                    
                    if grid_forest:
                        forest_values.append(np.median(grid_forest))
                    if grid_distances:
                        distance_values.append(np.median(grid_distances))
        
        # Calculate median forest quality and service distance
        if forest_values:
            self.avg_forest_quality = np.median(forest_values)
        else:
            self.avg_forest_quality = 0
            
        if distance_values:
            self.avg_distance_to_service = np.median(distance_values)
        else:
            self.avg_distance_to_service = 0
    
    def locate_residents(self):
        # Place specified number of residents in each step
        for _ in range(self.residentsperstep):
            resident = ResidentAgent(self.next_id(), self)
            
            # Set resident's weights and valuation
            resident.weightService = self.forest_gradient_service
            resident.weightForest = 2 - self.forest_gradient_service
            
            # Evaluate and find the best position
            best_pos, best_utility = self.evaluate_best_position(resident)
            
            # Place agent on the grid
            if best_pos:
                self.grid.place_agent(resident, best_pos)
                resident.utility = best_utility
                
                # Update forest value at this location
                x, y = best_pos
                self.forest[x, y] = 0
                
                # If changing nearby forest is enabled
                if self.change_forest_nearby:
                    # Update surrounding forest values
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                if dx == 0 and dy == 0:
                                    continue
                                elif abs(dx) + abs(dy) <= 1:  # Four cardinal directions
                                    self.forest[nx, ny] *= 0.7
                                else:  # Diagonal directions
                                    self.forest[nx, ny] *= 0.9
                
                # Record last resident's position for service center placement
                self.selx, self.sely = best_pos
            
            # Update counter
            self.counter += 1
    
    def evaluate_best_position(self, resident):
        # Evaluate and select the best position
        xlist = []
        ylist = []
        util_list = []
        
        # Try positions randomly
        attempts = 0
        while attempts < self.market_knowledge:
            # Choose a random position
            canx = random.randint(0, self.width - 1)
            cany = random.randint(0, self.height - 1)
            
            # Check if position is already occupied
            cell_contents = self.grid.get_cell_list_contents((canx, cany))
            if cell_contents:
                continue  # If occupied, try another position
            
            attempts += 1
            
            # Calculate utility
            forest_val = self.forest[canx, cany]
            sddist_val = self.sddist[canx, cany]
            
            # Prevent division by zero
            if sddist_val <= 0:
                sddist_val = 0.01
            
            # Calculate utility
            try:
                forest_component = forest_val ** resident.weightForest
                distance_component = (1 / sddist_val) ** resident.weightService
                utility = distance_component * forest_component
                
                # Save valid position and its utility
                xlist.append(canx)
                ylist.append(cany)
                util_list.append(utility)
            except:
                # Handle invalid values
                continue
        
        # Find best position
        if util_list:
            max_util = max(util_list)
            max_pos = util_list.index(max_util)
            return (xlist[max_pos], ylist[max_pos]), max_util
        
        # If no valid position found, try to find an empty position
        for x in range(self.width):
            for y in range(self.height):
                cell_contents = self.grid.get_cell_list_contents((x, y))
                if not cell_contents:
                    return (x, y), 0
        
        # If all positions are occupied, return (0,0) and print warning
        print("Warning: All positions are occupied, cannot place new resident")
        return (0, 0), 0
    
    def locate_service(self):
        # Create new service center
        service = ServiceAgent(self.next_id(), self)
        
        # Try to place near the last resident
        if hasattr(self, 'selx') and hasattr(self, 'sely'):
            x, y = self.selx, self.sely
            
            # If cell is already occupied, randomly search nearby empty cells
            cell_contents = self.grid.get_cell_list_contents((x, y))
            if cell_contents:
                attempts = 20  # Limit attempts
                while attempts > 0:
                    # Random direction
                    angle = random.uniform(0, 2 * math.pi)
                    distance = 1
                    
                    # Calculate new position
                    nx = int(x + distance * math.cos(angle))
                    ny = int(y + distance * math.sin(angle))
                    
                    # Ensure within grid
                    nx = max(0, min(nx, self.width - 1))
                    ny = max(0, min(ny, self.height - 1))
                    
                    # Check if new position is empty
                    new_cell_contents = self.grid.get_cell_list_contents((nx, ny))
                    if not new_cell_contents:
                        x, y = nx, ny
                        break
                    
                    attempts -= 1
        else:
            # Default place in center
            x, y = self.width // 2, self.height // 2
        
        # Place service center
        self.grid.place_agent(service, (x, y))
        
        # Update forest value at this location
        self.forest[x, y] = 0
        
        # If changing nearby forest is enabled
        if self.change_forest_nearby:
            # Update surrounding forest values
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if dx == 0 and dy == 0:
                            continue
                        elif abs(dx) + abs(dy) <= 1:  # Four cardinal directions
                            self.forest[nx, ny] *= 0.5
                        else:  # Diagonal directions
                            self.forest[nx, ny] *= 0.7
        
        # Update service distances
        self.update_service_distances()
        
        # Reset counter
        self.counter = 0
    
    def step(self):
        # One time step of the model
        self.locate_residents()
        
        # If counter reaches threshold, place new service center
        if self.counter > self.residentsperservice:
            self.locate_service()
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Export grid data (every step)
        self.export_grid_data(self.experiment_type)
        
        # Collect data
        self.datacollector.collect(self)
        
        # Export data - ensure export every step, pass experiment type
        self.export_data_tick(self.experiment_type)
        
        # Increment step counter
        self.steps += 1
        
        # Check stop conditions
        residents_count = len([a for a in self.agents if isinstance(a, ResidentAgent)])
        outside_residents = self.count_outside_residents()
        
        if residents_count >= 1600 or outside_residents > self.allowed_outside_houses:
            self.running = False

# Main function: Run model
def run_model(width=50, height=50, 
              spatial_resolution=1, 
              forest_gradient_service=1, 
              market_knowledge=10, 
              allowed_outside_houses=10, 
              city_radius=20, 
              smoothness=10, 
              change_forest_nearby=True,
              max_steps=1000,
              export_path="C:/Users/User/OneDrive - Auburn University/PHD/ABM spatial resolution/PYTHON/data/",
              experiment_type="baseline"):
    
    # Ensure output directory exists
    os.makedirs(export_path, exist_ok=True)
    print(f"Running experiment: {experiment_type}")
    print(f"Parameters: sr={spatial_resolution}, fgs={forest_gradient_service}, mk={market_knowledge}, aoh={allowed_outside_houses}, cr={city_radius}, s={smoothness}, cfn={change_forest_nearby}")
    print(f"Data will be saved to: {os.path.abspath(export_path)}")
    
    # Create model, pass experiment type
    model = UrbanGrowthModel(
        width=width,
        height=height,
        spatial_resolution=spatial_resolution,
        forest_gradient_service=forest_gradient_service,
        market_knowledge=market_knowledge,
        allowed_outside_houses=allowed_outside_houses,
        city_radius=city_radius,
        smoothness=smoothness,
        change_forest_nearby=change_forest_nearby,
        export_path=export_path,
        experiment_type=experiment_type  # Pass experiment type
    )
    
    # Run model until stop conditions are met or max steps reached
    for i in range(max_steps):
        if not model.running:
            print(f"Model stopped at step {i+1}")
            break
        
        # Export data at each step, experiment type already set in model
        model.step()
        
        # Print status every 20 steps
        if i % 20 == 0:
            residents_count = len([a for a in model.agents if isinstance(a, ResidentAgent)])
            print(f"Step {i}: Total residents = {residents_count}, experiment type = {experiment_type}")
    
    # Export data one last time
    model.export_data_tick(experiment_type)
    
    residents_count = len([a for a in model.agents if isinstance(a, ResidentAgent)])
    print(f"Experiment {experiment_type} completed, final resident count: {residents_count}")
    csv_file = f"{export_path}abm_data_{spatial_resolution}x{spatial_resolution}_{experiment_type}.csv"
    print(f"Data exported to: {os.path.abspath(csv_file)}")
    
    # Return collected data
    return model.datacollector.get_model_vars_dataframe()

# Run multiple experiments
def run_multiple_experiments(spatial_resolution=1, export_path="C:/Users/User/OneDrive - Auburn University/PHD/ABM spatial resolution/PYTHON/data/"):
    """
    Run multiple experimental conditions at once
    
    Parameters:
    spatial_resolution -- spatial resolution (1, 2, 4, 8 etc.)
    export_path -- output directory
    """
    # Experiment parameter list, add more experiment types
    experiments = [
        {"forest_gradient_service": 0.5, "market_knowledge": 5, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "service_market_low"},
        {"forest_gradient_service": 0.5, "market_knowledge": 15, "allowed_outside_houses": 50, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "outsideHouses_service_low"},
        {"forest_gradient_service": 0.5, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 5, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "cityRadius_service_low"},
        {"forest_gradient_service": 0.5, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 5, "change_forest_nearby": False, "experiment_type": "service_smoothness_low"},
        {"forest_gradient_service": 0.5, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "forest-gradient-low"},
        {"forest_gradient_service": 0.5, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "changeForest_service_low"},
        {"forest_gradient_service": 1, "market_knowledge": 5, "allowed_outside_houses": 50, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "outsideHouses_market_low"},
        {"forest_gradient_service": 1, "market_knowledge": 5, "allowed_outside_houses": 100, "city_radius": 5, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "cityRadius_market_low"},
        {"forest_gradient_service": 1, "market_knowledge": 5, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 5, "change_forest_nearby": False, "experiment_type": "market_smoothness_low"},
        {"forest_gradient_service": 1, "market_knowledge": 5, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "market-knowledge-low"},
        {"forest_gradient_service": 1, "market_knowledge": 5, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "changeForest_market_low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 50, "city_radius": 5, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "outsideHouses_cityRadius_low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 50, "city_radius": 25, "smoothness": 5, "change_forest_nearby": False, "experiment_type": "outsideHouses_smoothness_low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 50, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "allowed-ousidehouse-low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 50, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "changeForest_outsideHouses_low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 5, "smoothness": 5, "change_forest_nearby": False, "experiment_type": "cityRadius_smoothness_low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 5, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "city-radius-low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 5, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "changeForest_cityRadius_low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 5, "change_forest_nearby": False, "experiment_type": "smoothness-low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 5, "change_forest_nearby": False, "experiment_type": "changeForest_smoothness_low"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "baseline"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": True, "experiment_type": "change-forest-on"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 30, "change_forest_nearby": False, "experiment_type": "smoothness-high"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 30, "change_forest_nearby": True, "experiment_type": "changeForest_smoothness_high"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 50, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "city-radius-high"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 50, "smoothness": 15, "change_forest_nearby": True, "experiment_type": "changeForest_cityRadius_high"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 50, "smoothness": 30, "change_forest_nearby": False, "experiment_type": "cityRadius_smoothness_high"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 1000, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "allowed-ousidehouse-high"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 1000, "city_radius": 25, "smoothness": 15, "change_forest_nearby": True, "experiment_type": "changeForest_outsideHouses_high"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 1000, "city_radius": 25, "smoothness": 30, "change_forest_nearby": False, "experiment_type": "outsideHouses_smoothness_high"},
        {"forest_gradient_service": 1, "market_knowledge": 15, "allowed_outside_houses": 1000, "city_radius": 50, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "outsideHouses_cityRadius_high"},
        {"forest_gradient_service": 1, "market_knowledge": 30, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "market-knowledge-high"},
        {"forest_gradient_service": 1, "market_knowledge": 30, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": True, "experiment_type": "changeForest_market_high"},
        {"forest_gradient_service": 1, "market_knowledge": 30, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 30, "change_forest_nearby": False, "experiment_type": "market_smoothness_high"},
        {"forest_gradient_service": 1, "market_knowledge": 30, "allowed_outside_houses": 100, "city_radius": 50, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "cityRadius_market_high"},
        {"forest_gradient_service": 1, "market_knowledge": 30, "allowed_outside_houses": 1000, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "outsideHouses_market_high"},
        {"forest_gradient_service": 2, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "forest-gradient-high"},
        {"forest_gradient_service": 2, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": True, "experiment_type": "changeForest_service_high"},
        {"forest_gradient_service": 2, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 30, "change_forest_nearby": False, "experiment_type": "service_smoothness_high"},
        {"forest_gradient_service": 2, "market_knowledge": 15, "allowed_outside_houses": 100, "city_radius": 50, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "cityRadius_service_high"},
        {"forest_gradient_service": 2, "market_knowledge": 15, "allowed_outside_houses": 1000, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "outsideHouses_service_high"},
        {"forest_gradient_service": 2, "market_knowledge": 30, "allowed_outside_houses": 100, "city_radius": 25, "smoothness": 15, "change_forest_nearby": False, "experiment_type": "service_market_high"}
    ]

    
# Record start time
    start_time = datetime.now()
    print(f"Starting batch experiments, spatial resolution is {spatial_resolution}, with {len(experiments)} experimental conditions")
    print(f"Start time: {start_time}")
    
    # Create summary file
    summary_file = f"{export_path}experiments_summary_{spatial_resolution}x{spatial_resolution}.csv"
    summary_headers = [
        "experiment_type", "spatial_resolution", "forest_gradient_service", 
        "market_knowledge", "allowed_outside_houses", "city_radius", 
        "smoothness", "change_forest_nearby", "total_steps", 
        "final_residents", "final_outside_residents", "final_forest_coverage"
    ]
    
    # Initialize summary file
    if not os.path.exists(summary_file):
        summary_df = pd.DataFrame(columns=summary_headers)
        summary_df.to_csv(summary_file, index=False)
    
    # Run each experiment
    for i, exp in enumerate(experiments):
        print(f"\nRunning experiment {i+1}/{len(experiments)}: {exp['experiment_type']}")
        
        try:
            # Run model
            run_model(
                width=50,
                height=50,
                spatial_resolution=spatial_resolution,
                forest_gradient_service=exp["forest_gradient_service"],
                market_knowledge=exp["market_knowledge"],
                allowed_outside_houses=exp["allowed_outside_houses"],
                city_radius=exp["city_radius"],
                smoothness=exp["smoothness"],
                change_forest_nearby=exp["change_forest_nearby"],
                max_steps=1000,  # Maximum steps can be adjusted
                export_path=export_path,
                experiment_type=exp["experiment_type"]
            )
            
            # Read last line from generated CSV file as result
            result_file = f"{export_path}abm_data_{spatial_resolution}x{spatial_resolution}_{exp['experiment_type']}.csv"
            if os.path.exists(result_file):
                try:
                    df = pd.read_csv(result_file)
                    if not df.empty:
                        last_row = df.iloc[-1]
                        
                        # Add to summary file
                        summary_row = {
                            "experiment_type": exp["experiment_type"],
                            "spatial_resolution": spatial_resolution,
                            "forest_gradient_service": exp["forest_gradient_service"],
                            "market_knowledge": exp["market_knowledge"],
                            "allowed_outside_houses": exp["allowed_outside_houses"],
                            "city_radius": exp["city_radius"],
                            "smoothness": exp["smoothness"],
                            "change_forest_nearby": exp["change_forest_nearby"],
                            "total_steps": last_row["tick"],
                            "final_residents": last_row["Plot_Houses"],
                            "final_outside_residents": last_row["Development_Beyond_City_Boundary"],
                            "final_forest_coverage": last_row["Plot_Forest"]
                        }
                        
                        # Add to summary file
                        summary_df = pd.DataFrame([summary_row])
                        summary_df.to_csv(summary_file, mode='a', header=False, index=False)
                        print(f"  Experiment results added to summary file: {summary_file}")
                        
                except Exception as e:
                    print(f"  Error processing result file: {e}")
        except Exception as e:
            print(f"  Experiment failed: {e}")
    
    # Record end time
    end_time = datetime.now()
    duration = end_time - start_time
    print("\nBatch experiments completed!")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total duration: {duration}")
    print(f"Experiment summary file: {os.path.abspath(summary_file)}")

    # Example usage
    if __name__ == "__main__":
        # Set output path
        export_path = "C:/Users/User/OneDrive - Auburn University/PHD/ABM spatial resolution/PYTHON/data/"
        
        # Ensure path exists
        os.makedirs(export_path, exist_ok=True)
        
        # Spatial resolution selection
        spatial_resolution = 200  # Can be set to different values
        
        # Choose run mode: single experiment, multiple experiments, or batch run
        
        # Option 1: Run single experiment
        # print("Running single model...")
        # run_model(
        #     spatial_resolution=spatial_resolution,
        #     forest_gradient_service=1, 
        #     market_knowledge=15,
        #     allowed_outside_houses=100,
        #     city_radius=25,
        #     smoothness=15,
        #     change_forest_nearby=False,
        #     export_path=export_path,
        #     experiment_type="baseline"
        # )
        
        # Option 2: Run all experiment conditions
        print("Running all experiment conditions...")
        run_multiple_experiments(
            spatial_resolution=spatial_resolution,
            export_path=export_path
        )