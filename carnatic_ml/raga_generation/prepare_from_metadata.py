import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from jsonschema import validate, ValidationError

class RagaMetadataProcessor:
    """
    Process and validate raga metadata files, preparing them for use in the 
    raga generation pipeline.
    """
    def __init__(self, metadata_dir: str, schema_path: str):
        """
        Initialize the RagaMetadataProcessor.
        
        Args:
            metadata_dir: Directory containing raga metadata JSON files
            schema_path: Path to the raga schema JSON file
        """
        self.metadata_dir = Path(metadata_dir)
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.raga_data: Dict[str, dict] = {}

    def _load_schema(self) -> dict:
        """Load and return the raga metadata schema"""
        with open(self.schema_path, 'r') as f:
            return json.load(f)

    def load_raga_metadata(self, raga_id: Optional[str] = None) -> Dict[str, dict]:
        """
        Load raga metadata files. If raga_id is provided, load only that raga.
        Otherwise, load all ragas in the metadata directory.
        
        Args:
            raga_id: Optional specific raga to load
            
        Returns:
            Dictionary mapping raga IDs to their metadata
        """
        if raga_id:
            file_path = self.metadata_dir / f"{raga_id}.json"
            if not file_path.exists():
                raise FileNotFoundError(f"No metadata found for raga: {raga_id}")
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.raga_data[raga_id] = data
        else:
            for file_path in self.metadata_dir.glob("*.json"):
                raga_id = file_path.stem
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.raga_data[raga_id] = data

        return self.raga_data

    def validate_raga_metadata(self, raga_id: str) -> bool:
        """
        Validate a specific raga's metadata against the schema.
        
        Args:
            raga_id: ID of the raga to validate
            
        Returns:
            True if valid, raises ValidationError if invalid
        """
        if raga_id not in self.raga_data:
            raise KeyError(f"Raga {raga_id} not loaded")

        data = self.raga_data[raga_id]
        
        try:
            # Validate against JSON schema
            validate(instance=data, schema=self.schema)
            
            # Additional validation of swara content
            self._validate_swara_content(raga_id, data)
            
            # Validate melakarta consistency if applicable
            self._validate_melakarta_consistency(raga_id, data)
            
            # Validate lakshana consistency
            self._validate_lakshana_consistency(raga_id, data)
            
            return True
            
        except ValidationError as e:
            raise ValidationError(f"Schema validation failed for raga {raga_id}: {str(e)}")

    def _validate_swara_content(self, raga_id: str, data: dict) -> None:
        """
        Validate swara content in arohanam/avarohanam and constraints.
        
        Args:
            raga_id: ID of the raga being validated
            data: Raga metadata dictionary
            
        Raises:
            ValidationError if validation fails
        """
        valid_swaras = {'S', 'R1', 'R2', 'R3', 'G1', 'G2', 'G3', 'M1', 'M2', 
                       'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3'}
        
        # Validate arohanam/avarohanam
        for swara in data['arohanam'] + data['avarohanam']:
            if swara not in valid_swaras:
                raise ValidationError(f"Invalid swara {swara} in raga {raga_id}")
        
        # Validate lakshana swaras
        lakshana = data.get('raga_lakshana', {})
        for field in ['vadi_swara', 'samvadi_swara', 'graha_swara', 'amsa_swara']:
            if lakshana.get(field) and lakshana[field] not in valid_swaras:
                raise ValidationError(
                    f"Invalid swara in {field}: {lakshana[field]} for raga {raga_id}"
                )
        
        # Validate swara lists in lakshana
        for field in ['jeeva_swaras', 'nyasa_swaras', 'varjya_swaras', 'vakra_swaras']:
            if field in lakshana:
                for swara in lakshana[field]:
                    if swara not in valid_swaras:
                        raise ValidationError(
                            f"Invalid swara in {field}: {swara} for raga {raga_id}"
                        )

    def _validate_melakarta_consistency(self, raga_id: str, data: dict) -> None:
        """
        Validate consistency of melakarta-related information.
        
        Args:
            raga_id: ID of the raga being validated
            data: Raga metadata dictionary
            
        Raises:
            ValidationError if validation fails
        """
        if data.get('is_melakarta'):
            if not data.get('melakarta_number'):
                raise ValidationError(
                    f"Melakarta raga {raga_id} missing melakarta number"
                )
            if data.get('parent_melakarta'):
                raise ValidationError(
                    f"Melakarta raga {raga_id} should not have parent_melakarta"
                )
        elif data.get('parent_melakarta'):
            if data.get('melakarta_number'):
                raise ValidationError(
                    f"Janya raga {raga_id} should not have melakarta_number"
                )

    def _validate_lakshana_consistency(self, raga_id: str, data: dict) -> None:
        """
        Validate consistency of raga lakshana information.
        
        Args:
            raga_id: ID of the raga being validated
            data: Raga metadata dictionary
            
        Raises:
            ValidationError if validation fails
        """
        lakshana = data.get('raga_lakshana', {})
        
        # All swaras in vadi, samvadi, etc. should be in arohanam/avarohanam
        allowed_swaras = set(data['arohanam'] + data['avarohanam'])
        
        for field in ['vadi_swara', 'samvadi_swara', 'graha_swara', 'amsa_swara']:
            if lakshana.get(field) and lakshana[field] not in allowed_swaras:
                raise ValidationError(
                    f"{field} {lakshana[field]} not in arohanam/avarohanam for raga {raga_id}"
                )
        
        # Validate swara lists against arohanam/avarohanam
        for field in ['jeeva_swaras', 'nyasa_swaras']:
            if field in lakshana:
                for swara in lakshana[field]:
                    if swara not in allowed_swaras:
                        raise ValidationError(
                            f"{field} swara {swara} not in arohanam/avarohanam for raga {raga_id}"
                        )

        # Varjya swaras should NOT be in arohanam/avarohanam
        if 'varjya_swaras' in lakshana:
            for swara in lakshana['varjya_swaras']:
                if swara in allowed_swaras:
                    raise ValidationError(
                        f"Varjya swara {swara} found in arohanam/avarohanam for raga {raga_id}"
                    )

    def get_raga_swara_constraints(self, raga_id: str) -> Dict[str, List[str]]:
        """
        Extract swara sequence constraints for a raga from its metadata.
        
        Args:
            raga_id: ID of the raga to process
            
        Returns:
            Dictionary containing:
                - allowed_swaras: List of swaras that can be used
                - vakra_patterns: List of special movement patterns 
                - arohanam: Ascending scale pattern
                - avarohanam: Descending scale pattern
        """
        if raga_id not in self.raga_data:
            raise KeyError(f"Raga {raga_id} not loaded")

        data = self.raga_data[raga_id]
        all_swaras = ['S', 'R1', 'R2', 'R3', 'G1', 'G2', 'G3', 'M1', 'M2', 
                      'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3']
        
        # Get allowed swaras by removing varjya swaras
        varjya = data['raga_lakshana'].get('varjya_swaras', [])
        allowed_swaras = [s for s in all_swaras if s not in varjya]

        # Get special movement patterns from vakra swaras
        vakra_patterns = data['raga_lakshana'].get('vakra_swaras', [])

        return {
            'allowed_swaras': allowed_swaras,
            'vakra_patterns': vakra_patterns,
            'arohanam': data['arohanam'],
            'avarohanam': data['avarohanam']
        }

    def get_raga_melodic_features(self, raga_id: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract important melodic features of a raga from its metadata.
        
        Args:
            raga_id: ID of the raga to process
            
        Returns:
            Dictionary containing important notes and their roles
        """
        if raga_id not in self.raga_data:
            raise KeyError(f"Raga {raga_id} not loaded")

        data = self.raga_data[raga_id]
        lakshana = data['raga_lakshana']

        return {
            'vadi': lakshana.get('vadi_swara', ''),
            'samvadi': lakshana.get('samvadi_swara', ''),
            'graha': lakshana.get('graha_swara', ''),
            'amsa': lakshana.get('amsa_swara', ''),
            'jeeva_swaras': lakshana.get('jeeva_swaras', []),
            'nyasa_swaras': lakshana.get('nyasa_swaras', []),
            'classification': data.get('raga_classification', {})
        }

if __name__ == "__main__":
    # Example usage
    metadata_dir = "../shared/ragas_metadata"
    schema_path = "../shared/schemas/raga_schema.json"
    
    processor = RagaMetadataProcessor(metadata_dir, schema_path)
    
    # Load a specific raga
    processor.load_raga_metadata("mohanam")
    
    # Validate the metadata
    try:
        processor.validate_raga_metadata("mohanam")
        print("Mohanam metadata is valid")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Get constraints for generation
    constraints = processor.get_raga_swara_constraints("mohanam")
    print(f"\nAllowed swaras for Mohanam: {constraints['allowed_swaras']}")
    print(f"Arohanam: {constraints['arohanam']}")
    print(f"Avarohanam: {constraints['avarohanam']}")
    
    # Get melodic features
    features = processor.get_raga_melodic_features("mohanam")
    print(f"\nImportant notes in Mohanam: {features}")
